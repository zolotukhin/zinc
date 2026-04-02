//! CLI entrypoints for configuring ZINC and starting local inference.
//! @section CLI & Entrypoints
//! This module wires together GPU initialization, model loading, tokenization,
//! and the single-process decode loop used for prompt-mode execution.
const builtin = @import("builtin");
const std = @import("std");
const gpu = @import("gpu/interface.zig");
const catalog_mod = @import("model/catalog.zig");
const diagnostics_mod = if (gpu.is_vulkan)
    @import("diagnostics.zig")
else if (gpu.is_metal)
    @import("diagnostics_metal.zig")
else
    struct {
        pub const ManagedModelInfo = struct {
            id: []const u8,
            display_name: []const u8,
            file_name: []const u8,
            size_bytes: u64,
            required_vram_bytes: u64,
            status_label: []const u8,
        };

        pub fn run(_: anytype, _: std.mem.Allocator) !void {
            return error.DiagnosticsUnsupportedOnThisBackend;
        }
    };
const gguf_mod = @import("model/gguf.zig");
const managed_mod = @import("model/managed.zig");
const tokenizer_mod = @import("model/tokenizer.zig");
const graph_mod = @import("compute/graph.zig");
const server_runtime = @import("server/runtime.zig");
// These modules import vulkan/ transitively — only available on Linux until T010-T014 refactor.
// On macOS they are stubbed out; the GPU abstraction refactor will make them platform-independent.
const loader_mod = if (gpu.is_vulkan) @import("model/loader.zig") else struct {};
const architecture_mod = if (gpu.is_vulkan) @import("model/architecture.zig") else struct {};
const forward_mod = if (gpu.is_vulkan) @import("compute/forward.zig") else struct {};

// Backend-specific imports (only one branch compiles per platform)
const instance_mod = if (gpu.is_vulkan) @import("vulkan/instance.zig") else gpu.backend;
const gpu_detect = if (gpu.is_vulkan) @import("vulkan/gpu_detect.zig") else struct {};
const http_mod = @import("server/http.zig");
const model_manager_mod = @import("server/model_manager_runtime.zig");
const routes_mod = @import("server/routes.zig");
const CommandPool = if (gpu.is_vulkan) @import("vulkan/command.zig").CommandPool else struct {
    pub fn init(_: anytype) !@This() {
        return .{};
    }
    pub fn deinit(_: *@This()) void {}
};
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
        _ = @import("model/catalog.zig");
        _ = @import("model/managed.zig");
        // These modules import vulkan/ directly — Vulkan-only until T010-T014 refactor
        _ = @import("compute/dmmv.zig");
        _ = @import("compute/elementwise.zig");
        _ = @import("compute/attention.zig");
        _ = @import("compute/forward.zig");
        _ = @import("model/loader.zig");
        _ = @import("model/architecture.zig");
        _ = @import("server/model_manager.zig");
        _ = @import("server/routes.zig");
    }
    if (gpu.is_metal) {
        _ = @import("metal/device.zig");
        _ = @import("metal/buffer.zig");
        _ = @import("metal/pipeline.zig");
        _ = @import("metal/command.zig");
        _ = @import("model/loader_metal.zig");
        _ = @import("compute/forward_metal.zig");
        _ = @import("server/model_manager_metal.zig");
        _ = @import("server/model_manager_runtime.zig");
        _ = @import("server/routes.zig");
    }
    // Platform-independent modules
    _ = @import("model/config.zig");
    _ = @import("model/gguf.zig");
    _ = @import("model/tokenizer.zig");
    _ = @import("compute/graph.zig");
    _ = @import("regression_tests.zig");
    _ = @import("server/http.zig");
    _ = @import("server/runtime.zig");
    _ = @import("server/session.zig");
    _ = @import("scheduler/request.zig");
    _ = @import("scheduler/scheduler.zig");
    _ = @import("scheduler/kv_cache.zig");
    if (gpu.is_vulkan) {
        _ = @import("diagnostics.zig");
    }
    if (gpu.is_metal) {
        _ = @import("diagnostics_metal.zig");
    }
}

/// Runtime configuration built from CLI flags and default values.
pub const Config = struct {
    /// Path to GGUF model file.
    model_path: ?[]const u8 = null,
    /// Managed model identifier from the built-in catalog/cache.
    model_id: ?[]const u8 = null,
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
    /// Maximum CLI decode tokens.
    max_tokens: u32 = 256,
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
    /// Show extended help including developer-only flags.
    show_help_all: bool = false,
    /// Run diagnostics and exit.
    check: bool = false,
    /// Optional model-management command.
    command: Command = .run,
    /// Positional model id for `zinc model ...`.
    command_model_id: ?[]const u8 = null,
    /// Force a managed-model command that would otherwise refuse.
    command_force: bool = false,
    /// Show unsupported catalog entries in `zinc model list`.
    show_all_models: bool = false,
};

pub const Command = enum {
    run,
    chat,
    model_list,
    model_pull,
    model_use,
    model_active,
    model_rm,
};

const ConnectionWorker = struct {
    conn: http_mod.Connection,
    manager: *model_manager_mod.ModelManager,
    server_state: *routes_mod.ServerState,

    fn run(self: *@This()) void {
        defer std.heap.page_allocator.destroy(self);
        defer self.conn.close();

        routes_mod.handleConnection(
            &self.conn,
            self.manager,
            self.server_state,
            std.heap.page_allocator,
        ) catch |err| {
            log.warn("Request failed: {s}", .{@errorName(err)});
        };
    }
};

fn runHttpServer(config: Config, manager: *model_manager_mod.ModelManager, allocator: std.mem.Allocator) void {
    if (config.profile) {
        if (manager.currentResources()) |resources| {
            if (comptime server_runtime.supports_runtime_profiling) {
                server_runtime.enableProfiling(&resources.engine) catch |err| {
                    log.warn("Failed to enable profiling: {s}", .{@errorName(err)});
                };
            } else {
                log.warn("Per-dispatch profiling is not available on the Metal HTTP server yet.", .{});
            }
        }
    }
    if (config.debug) {
        if (manager.currentResources()) |resources| {
            server_runtime.enableLogitsReadback(&resources.engine);
        }
    }

    var server = http_mod.Server.init(allocator, config.port) catch |err| {
        log.err("Failed to start HTTP server: {s}", .{@errorName(err)});
        std.process.exit(1);
    };
    defer server.deinit();
    log.info("Server listening on 0.0.0.0:{d}", .{config.port});
    if (config.command == .chat) {
        launchChatUi(config.port);
    }
    log.info("Press Ctrl+C to stop", .{});

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
    var poll_fds = [1]posix.pollfd{.{
        .fd = server.listener.stream.handle,
        .events = posix.POLL.IN,
        .revents = 0,
    }};

    while (!Handler.shutdown_requested) {
        poll_fds[0].revents = 0;
        const ready = posix.poll(&poll_fds, 100) catch |err| {
            log.warn("Listener poll failed: {s}", .{@errorName(err)});
            continue;
        };
        if (ready == 0) continue;

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
            .manager = manager,
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

fn launchChatUi(port: u16) void {
    var url_buf: [128]u8 = undefined;
    const url = std.fmt.bufPrint(&url_buf, "http://127.0.0.1:{d}/chat", .{port}) catch |err| {
        log.warn("Failed to format chat URL: {s}", .{@errorName(err)});
        return;
    };
    log.info("Opening chat UI at {s}", .{url});
    launchBrowser(url) catch |err| {
        log.warn("Failed to open browser for {s}: {s}", .{ url, @errorName(err) });
    };
}

fn launchBrowser(url: []const u8) !void {
    const argv: []const []const u8 = switch (builtin.os.tag) {
        .macos => &[_][]const u8{ "open", url },
        .linux => &[_][]const u8{ "xdg-open", url },
        .windows => &[_][]const u8{ "cmd", "/c", "start", "", url },
        else => return error.BrowserLaunchUnsupported,
    };

    var child = std.process.Child.init(argv, std.heap.page_allocator);
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Ignore;
    child.stderr_behavior = .Ignore;

    const term = try child.spawnAndWait();
    switch (term) {
        .Exited => |code| {
            if (code != 0) return error.BrowserLauncherFailed;
        },
        else => return error.BrowserLauncherFailed,
    }
}

const PreparedPrompt = struct {
    text: []const u8,
    owned_buf: ?[]u8 = null,

    fn deinit(self: *PreparedPrompt, allocator: std.mem.Allocator) void {
        if (self.owned_buf) |buf| allocator.free(buf);
        self.* = undefined;
    }
};

const ResolvedStartupModel = struct {
    spec: model_manager_mod.LoadSpec,
    owned_path: ?[]u8 = null,
    owned_managed_id: ?[]u8 = null,

    fn deinit(self: *ResolvedStartupModel, allocator: std.mem.Allocator) void {
        if (self.owned_path) |path| allocator.free(path);
        if (self.owned_managed_id) |id| allocator.free(id);
        self.* = undefined;
    }
};

const ResolvedCheckTarget = struct {
    model_path: ?[]const u8 = null,
    managed_model: ?diagnostics_mod.ManagedModelInfo = null,
    owned_path: ?[]u8 = null,

    fn deinit(self: *ResolvedCheckTarget, allocator: std.mem.Allocator) void {
        if (self.owned_path) |path| allocator.free(path);
        self.* = undefined;
    }
};

const banner =
    \\ZINC — Zig INferenCe Engine for AMD GPUs
    \\
    \\Usage:
    \\  zinc -m <model.gguf> --prompt "Hello"
    \\  zinc -m <model.gguf> [-p 8080]
    \\  zinc chat [-m <model.gguf> | --model-id <id>] [-p 9090]
    \\  zinc --model-id <id> [--prompt "Hello"]
    \\  zinc --check [-m <model.gguf> | --model-id <id>]
    \\  zinc model <list|pull|use|active|rm> [args]
    \\
    \\Common options:
    \\  -m, --model <path>       GGUF model file to load
    \\  --model-id <id>          Managed model id from the local catalog/cache
    \\  --prompt <text>          Run one prompt in CLI mode instead of starting the server
    \\  --chat                   Apply the model chat template to --prompt
    \\  -n, --max-tokens <n>     Max generated tokens in CLI mode (default: 256)
    \\  -d, --device <id>        Vulkan device index (default: 0)
    \\  -c, --context <size>     Context length (default: 4096)
    \\  --kv-quant <bits>        TurboQuant KV cache bits: 0/2/3/4 (default: 0)
    \\
    \\Server options:
    \\  -p, --port <port>        Server port (default: 8080)
    \\  --parallel <n>           Max concurrent requests (default: 4)
    \\  chat                     Start the server on port 9090 and open the built-in chat UI in your browser
    \\
    \\Model management:
    \\  model list [--all]       List managed models for the detected GPU
    \\  model pull <id>          Download a supported managed model into the local cache
    \\  model use <id>           Set the active managed model for future runs
    \\  model active             Print the active managed model
    \\  model rm [-f] <id>       Remove a cached managed model; -f unloads it first if active
    \\
    \\Diagnostics:
    \\  --check                  Run system diagnostics and verify dependencies
    \\  -h, --help               Show this help
    \\  --help-all               Show diagnostics and developer-only flags too
    \\
    \\Use `--help-all` to show graph export, profiling, and debug flags.
    \\
;

const banner_full =
    \\ZINC — Zig INferenCe Engine for AMD GPUs
    \\
    \\Usage:
    \\  zinc -m <model.gguf> --prompt "Hello"
    \\  zinc -m <model.gguf> [-p 8080]
    \\  zinc chat [-m <model.gguf> | --model-id <id>] [-p 9090]
    \\  zinc --model-id <id> [--prompt "Hello"]
    \\  zinc --check [-m <model.gguf> | --model-id <id>]
    \\  zinc model <list|pull|use|active|rm> [args]
    \\
    \\Common options:
    \\  -m, --model <path>       GGUF model file to load
    \\  --model-id <id>          Managed model id from the local catalog/cache
    \\  --prompt <text>          Run one prompt in CLI mode instead of starting the server
    \\  --chat                   Apply the model chat template to --prompt
    \\  -n, --max-tokens <n>     Max generated tokens in CLI mode (default: 256)
    \\  -d, --device <id>        Vulkan device index (default: 0)
    \\  -c, --context <size>     Context length (default: 4096)
    \\  --kv-quant <bits>        TurboQuant KV cache bits: 0/2/3/4 (default: 0)
    \\
    \\Server options:
    \\  -p, --port <port>        Server port (default: 8080)
    \\  --parallel <n>           Max concurrent requests (default: 4)
    \\  chat                     Start the server on port 9090 and open the built-in chat UI in your browser
    \\
    \\Model management:
    \\  model list [--all]       List managed models for the detected GPU
    \\  model pull <id>          Download a supported managed model into the local cache
    \\  model use <id>           Set the active managed model for future runs
    \\  model active             Print the active managed model
    \\  model rm [-f] <id>       Remove a cached managed model; -f unloads it first if active
    \\
    \\Diagnostics:
    \\  --check                  Run system diagnostics and verify dependencies
    \\
    \\Analysis and developer options:
    \\  --graph-report <path>    Write decode-graph analysis JSON report
    \\  --graph-dot <path>       Write decode-graph Graphviz DOT from GGUF metadata
    \\  --profile                Enable runtime profiling (per-dispatch on Vulkan, phase summary on Metal)
    \\  --debug                  Enable verbose debug logging
    \\
    \\Help:
    \\  -h, --help               Show the short help
    \\  --help-all               Show the full help
    \\
;

fn helpText(show_all: bool) []const u8 {
    return if (show_all) banner_full else banner;
}

/// Parse the process argument vector into a validated runtime configuration.
/// @param args Raw argv slice, including argv[0].
/// @returns A populated Config value or a validation error describing the first invalid flag.
pub fn parseArgs(args: []const [:0]const u8) !Config {
    var config = Config{};
    var port_explicit = false;
    var i: usize = 1; // skip argv[0]

    while (i < args.len) : (i += 1) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            config.show_help = true;
            return config;
        } else if (std.mem.eql(u8, arg, "--help-all")) {
            config.show_help = true;
            config.show_help_all = true;
            return config;
        } else if (std.mem.eql(u8, arg, "chat")) {
            if (config.command != .run) return error.UnknownArgument;
            config.command = .chat;
            if (!port_explicit) config.port = 9090;
        } else if (std.mem.eql(u8, arg, "model")) {
            if (config.command != .run) return error.UnknownArgument;
            i += 1;
            if (i >= args.len) return error.MissingModelSubcommand;
            const sub = args[i];
            if (std.mem.eql(u8, sub, "list")) {
                config.command = .model_list;
            } else if (std.mem.eql(u8, sub, "pull")) {
                config.command = .model_pull;
                i += 1;
                if (i >= args.len) return error.MissingArgValue;
                config.command_model_id = args[i];
            } else if (std.mem.eql(u8, sub, "use")) {
                config.command = .model_use;
                i += 1;
                if (i >= args.len) return error.MissingArgValue;
                config.command_model_id = args[i];
            } else if (std.mem.eql(u8, sub, "active")) {
                config.command = .model_active;
            } else if (std.mem.eql(u8, sub, "rm") or std.mem.eql(u8, sub, "remove")) {
                config.command = .model_rm;
                while (i + 1 < args.len and (std.mem.eql(u8, args[i + 1], "-f") or std.mem.eql(u8, args[i + 1], "--force"))) : (i += 1) {
                    config.command_force = true;
                }
                i += 1;
                if (i >= args.len) return error.MissingArgValue;
                config.command_model_id = args[i];
            } else {
                return error.UnknownArgument;
            }
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.model_path = args[i];
        } else if (std.mem.eql(u8, arg, "--model-id")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.model_id = args[i];
        } else if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--port")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.port = std.fmt.parseInt(u16, args[i], 10) catch return error.InvalidPort;
            port_explicit = true;
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
        } else if (std.mem.eql(u8, arg, "-n") or std.mem.eql(u8, arg, "--max-tokens")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.max_tokens = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidMaxTokens;
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
        } else if (std.mem.eql(u8, arg, "--max-tokens") or std.mem.eql(u8, arg, "-n")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.max_tokens = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidMaxTokens;
        } else if (std.mem.eql(u8, arg, "--profile")) {
            config.profile = true;
        } else if (std.mem.eql(u8, arg, "--debug")) {
            config.debug = true;
        } else if (std.mem.eql(u8, arg, "--check")) {
            config.check = true;
        } else if (std.mem.eql(u8, arg, "--all")) {
            config.show_all_models = true;
        } else if (std.mem.eql(u8, arg, "-f") or std.mem.eql(u8, arg, "--force")) {
            config.command_force = true;
        } else {
            return error.UnknownArgument;
        }
    }

    if (config.command == .chat and config.prompt != null) {
        return error.ChatCommandDoesNotTakePrompt;
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

fn resolveStartupModel(config: Config, allocator: std.mem.Allocator) !ResolvedStartupModel {
    if (config.model_id) |model_id| {
        const path = try managed_mod.resolveInstalledModelPath(model_id, allocator);
        const model_id_copy = try allocator.dupe(u8, model_id);
        return .{
            .spec = .{ .model_path = path, .managed_id = model_id_copy },
            .owned_path = path,
            .owned_managed_id = model_id_copy,
        };
    }

    if (config.model_path) |model_path| {
        return .{ .spec = .{ .model_path = model_path } };
    }

    const active = try managed_mod.readActiveSelection(allocator);
    if (active) |selection| {
        const path = try managed_mod.resolveInstalledModelPath(selection.model_id, allocator);
        return .{
            .spec = .{ .model_path = path, .managed_id = selection.model_id },
            .owned_path = path,
            .owned_managed_id = selection.model_id,
        };
    }

    return error.NoModelSpecified;
}

fn resolveCheckTarget(config: Config, allocator: std.mem.Allocator) !ResolvedCheckTarget {
    if (config.model_id) |model_id| {
        const entry = catalog_mod.find(model_id) orelse return error.UnknownManagedModel;

        var resolved = ResolvedCheckTarget{
            .managed_model = .{
                .id = entry.id,
                .display_name = entry.display_name,
                .file_name = entry.file_name,
                .size_bytes = entry.size_bytes,
                .required_vram_bytes = entry.required_vram_bytes,
                .status_label = @tagName(entry.status),
            },
        };

        if (managed_mod.isInstalled(model_id, allocator)) {
            const path = try managed_mod.resolveInstalledModelPath(model_id, allocator);
            resolved.model_path = path;
            resolved.owned_path = path;
        }

        return resolved;
    }

    if (config.model_path) |model_path| {
        return .{ .model_path = model_path };
    }

    return .{};
}

const ManagedGpuSupport = struct {
    profile: []u8,
    vram_budget_bytes: u64,
    from_cache: bool,

    fn deinit(self: *ManagedGpuSupport, allocator: std.mem.Allocator) void {
        allocator.free(self.profile);
        self.* = undefined;
    }
};

fn resolveManagedGpuSupport(device_index: u32, allocator: std.mem.Allocator) !ManagedGpuSupport {
    if (try managed_mod.readCachedGpuProfile(device_index, allocator)) |cached| {
        defer {
            var owned = cached;
            owned.deinit(allocator);
        }
        return .{
            .profile = try allocator.dupe(u8, cached.profile),
            .vram_budget_bytes = cached.vram_budget_bytes,
            .from_cache = true,
        };
    }

    if (gpu.is_vulkan) {
        var vk_instance = try instance_mod.Instance.init(allocator, device_index);
        defer vk_instance.deinit();

        const gpu_config = gpu_detect.detect(&vk_instance);
        const profile = catalog_mod.profileForGpu(gpu_config);
        const vram_budget_bytes = vk_instance.vramBytes();

        try managed_mod.writeCachedGpuProfile(device_index, profile, gpu_config.nameSlice(), vram_budget_bytes, allocator);

        return .{
            .profile = try allocator.dupe(u8, profile),
            .vram_budget_bytes = vram_budget_bytes,
            .from_cache = false,
        };
    }

    if (gpu.is_metal) {
        const metal_device_mod = @import("metal/device.zig");

        var device = try metal_device_mod.MetalDevice.init(allocator, device_index);
        defer device.deinit();

        const profile = catalog_mod.profileForMetal();
        const vram_budget_bytes = blk: {
            const working_set = device.recommendedMaxWorkingSetSize();
            break :blk if (working_set > 0) working_set else device.totalMemory();
        };
        const device_name = @tagName(device.chip);

        try managed_mod.writeCachedGpuProfile(device_index, profile, device_name, vram_budget_bytes, allocator);

        return .{
            .profile = try allocator.dupe(u8, profile),
            .vram_budget_bytes = vram_budget_bytes,
            .from_cache = false,
        };
    }

    return error.GpuDetectionUnavailable;
}

fn printManagedModelList(config: Config, allocator: std.mem.Allocator) !void {
    var active = try managed_mod.readActiveSelection(allocator);
    defer if (active) |*selection| selection.deinit(allocator);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);
    const active_model_id = if (active) |selection| selection.model_id else null;
    const backend_name = if (gpu.is_metal) "Metal" else if (gpu.is_vulkan) "Vulkan" else "GPU";

    const support = resolveManagedGpuSupport(config.device_index, allocator) catch |err| {
        if (!config.show_all_models) {
            var stderr_buffer: [1024]u8 = undefined;
            var stderr = std.fs.File.stderr().writerStreaming(&stderr_buffer);
            try stderr.interface.print("Unable to initialize {s} for GPU detection: {s}\n", .{ backend_name, @errorName(err) });
            try stderr.interface.writeAll("Use `zinc model list --all` to inspect the catalog without live fit checks.\n");
            try stderr.interface.flush();
            return error.GpuDetectionUnavailable;
        }

        try stdout.interface.print("{s} GPU detection unavailable ({s}). Showing the full catalog without live fit checks.\n\n", .{ backend_name, @errorName(err) });
        try stdout.interface.writeAll("ID                             Released     Status      Fit    Installed   Active   Notes\n");
        for (catalog_mod.entries) |entry| {
            const installed = managed_mod.isInstalled(entry.id, allocator);
            const is_active = active_model_id != null and std.mem.eql(u8, active_model_id.?, entry.id);
            try stdout.interface.print(
                "{s: <30} {s: <12} {s: <11} {s: <6} {s: <11} {s: <8} {s}\n",
                .{
                    entry.id,
                    entry.release_date,
                    "catalog",
                    "n/a",
                    if (installed) "yes" else "no",
                    if (is_active) "yes" else "no",
                    "fit unavailable without live GPU probe",
                },
            );
        }
        try stdout.interface.flush();
        return;
    };
    defer {
        var owned = support;
        owned.deinit(allocator);
    }

    try stdout.interface.print(
        "Detected GPU profile: {s}{s}\n\n",
        .{
            support.profile,
            if (support.from_cache) " (cached)" else "",
        },
    );
    try stdout.interface.writeAll("ID                             Released     Status      Fit    Installed   Active   Notes\n");

    var rendered_any = false;
    for (catalog_mod.entries) |entry| {
        const tested_profile_match = catalog_mod.supportsProfile(entry, support.profile);
        const installed = managed_mod.isInstalled(entry.id, allocator);
        const fit = managed_mod.describeFit(entry, support.vram_budget_bytes, allocator) catch managed_mod.ModelFit{
            .required_vram_bytes = entry.required_vram_bytes,
            .fits_current_gpu = catalog_mod.fitsGpu(entry, support.vram_budget_bytes),
            .exact = false,
        };
        const supported_now = tested_profile_match and fit.fits_current_gpu;
        if (!config.show_all_models and !supported_now) continue;

        rendered_any = true;
        const is_active = active_model_id != null and std.mem.eql(u8, active_model_id.?, entry.id);
        const status_label = if (supported_now)
            "supported"
        else if (tested_profile_match)
            "too-large"
        else
            "hidden";
        try stdout.interface.print(
            "{s: <30} {s: <12} {s: <11} {s: <6} {s: <11} {s: <8} {s}\n",
            .{
                entry.id,
                entry.release_date,
                status_label,
                if (fit.fits_current_gpu) "yes" else "no",
                if (installed) "yes" else "no",
                if (is_active) "yes" else "no",
                if (fit.exact) "tested + exact fit" else "tested + catalog fit",
            },
        );
    }

    if (!rendered_any) {
        try stdout.interface.writeAll("No managed models are currently marked supported and fitting for this GPU profile.\n");
    }

    try stdout.interface.flush();
}

const LocalAdminRemoveResponse = struct {
    status: u16,
    payload: []u8,
    body: []const u8,

    fn deinit(self: *LocalAdminRemoveResponse, allocator: std.mem.Allocator) void {
        allocator.free(self.payload);
        self.* = undefined;
    }
};

const ManagedRemoveOutcome = struct {
    unloaded_from_gpu: bool,
    cleared_active_selection: bool,
    deleted_model: bool,
    deleted_manifest: bool,
    removed_dir: bool,
};

fn tryRemoveManagedModelViaLocalServer(
    port: u16,
    model_id: []const u8,
    force: bool,
    allocator: std.mem.Allocator,
) !?LocalAdminRemoveResponse {
    const address = try std.net.Address.parseIp4("127.0.0.1", port);
    var stream = std.net.tcpConnectToAddress(address) catch return null;
    defer stream.close();

    const request_body = try std.fmt.allocPrint(
        allocator,
        "{{\"model\":\"{s}\",\"force\":{s}}}",
        .{ model_id, if (force) "true" else "false" },
    );
    defer allocator.free(request_body);

    const request = try std.fmt.allocPrint(
        allocator,
        "POST /v1/models/remove HTTP/1.1\r\nHost: 127.0.0.1:{d}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
        .{ port, request_body.len, request_body },
    );
    defer allocator.free(request);
    try stream.writeAll(request);

    var response: std.ArrayList(u8) = .{};
    defer response.deinit(allocator);

    var buf: [4096]u8 = undefined;
    while (true) {
        const n = try stream.read(&buf);
        if (n == 0) break;
        try response.appendSlice(allocator, buf[0..n]);
        if (response.items.len > 64 * 1024) return error.ResponseTooLarge;
    }

    const header_end = std.mem.indexOf(u8, response.items, "\r\n\r\n") orelse return null;
    const status = parseHttpStatus(response.items[0..header_end]) orelse return null;
    const payload = try allocator.dupe(u8, response.items);
    return .{
        .status = status,
        .payload = payload,
        .body = payload[header_end + 4 ..],
    };
}

fn parseHttpStatus(header: []const u8) ?u16 {
    const line_end = std.mem.indexOf(u8, header, "\r\n") orelse header.len;
    const line = header[0..line_end];
    const first_space = std.mem.indexOfScalar(u8, line, ' ') orelse return null;
    const rest = line[first_space + 1 ..];
    if (rest.len < 3) return null;
    return std.fmt.parseInt(u16, rest[0..3], 10) catch null;
}

fn jsonFieldIsTrue(body: []const u8, key: []const u8) bool {
    var needle_buf: [96]u8 = undefined;
    const compact = std.fmt.bufPrint(&needle_buf, "\"{s}\":true", .{key}) catch return false;
    if (std.mem.indexOf(u8, body, compact) != null) return true;
    const spaced = std.fmt.bufPrint(&needle_buf, "\"{s}\": true", .{key}) catch return false;
    return std.mem.indexOf(u8, body, spaced) != null;
}

fn extractJsonMessage(body: []const u8) ?[]const u8 {
    return extractJsonStringField(body, "message");
}

fn extractJsonStringField(body: []const u8, key: []const u8) ?[]const u8 {
    var needle_buf: [128]u8 = undefined;
    const compact = std.fmt.bufPrint(&needle_buf, "\"{s}\":\"", .{key}) catch return null;
    if (std.mem.indexOf(u8, body, compact)) |pos| {
        const start = pos + compact.len;
        return body[start .. start + (findJsonStringEnd(body[start..]) orelse return null)];
    }
    const spaced = std.fmt.bufPrint(&needle_buf, "\"{s}\": \"", .{key}) catch return null;
    if (std.mem.indexOf(u8, body, spaced)) |pos| {
        const start = pos + spaced.len;
        return body[start .. start + (findJsonStringEnd(body[start..]) orelse return null)];
    }
    return null;
}

fn findJsonStringEnd(s: []const u8) ?usize {
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (s[i] == '\\') {
            i += 1;
            continue;
        }
        if (s[i] == '"') return i;
    }
    return null;
}

fn printManagedRemoveSummary(model_id: []const u8, outcome: ManagedRemoveOutcome) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);
    if (outcome.unloaded_from_gpu) {
        try stdout.interface.print("Unloaded {s} from GPU memory\n", .{model_id});
    }
    if (outcome.deleted_model) {
        try stdout.interface.writeAll("Deleted: model.gguf\n");
    }
    if (outcome.deleted_manifest) {
        try stdout.interface.writeAll("Deleted: manifest.json\n");
    }
    if (outcome.removed_dir) {
        try stdout.interface.writeAll("Removed empty cache directory\n");
    }
    if (outcome.cleared_active_selection) {
        try stdout.interface.writeAll("Cleared active model selection\n");
    }
    try stdout.interface.print("Removed: {s}\n", .{model_id});
    try stdout.interface.flush();
}

fn printCommandError(message: []const u8) !void {
    var stderr_buffer: [1024]u8 = undefined;
    var stderr = std.fs.File.stderr().writerStreaming(&stderr_buffer);
    try stderr.interface.print("{s}\n", .{message});
    try stderr.interface.flush();
}

fn runModelCommand(config: Config, allocator: std.mem.Allocator) !void {
    switch (config.command) {
        .model_active => {
            var active = try managed_mod.readActiveSelection(allocator);
            defer if (active) |*selection| selection.deinit(allocator);

            var stdout_buffer: [1024]u8 = undefined;
            var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);
            if (active) |selection| {
                try stdout.interface.print("{s}\n", .{selection.model_id});
            } else {
                try stdout.interface.writeAll("No active managed model configured.\n");
            }
            try stdout.interface.flush();
        },
        .model_list => try printManagedModelList(config, allocator),
        .model_pull, .model_use => {
            const model_id = config.command_model_id orelse return error.MissingArgValue;
            const entry = catalog_mod.find(model_id) orelse return error.UnknownManagedModel;

            var support = try resolveManagedGpuSupport(config.device_index, allocator);
            defer support.deinit(allocator);

            if (!catalog_mod.supportsProfile(entry.*, support.profile)) return error.ModelUnsupportedOnThisGpu;

            if (config.command == .model_pull) {
                var stdout_buffer: [4096]u8 = undefined;
                var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);
                try managed_mod.pullModel(entry.*, allocator, &stdout.interface);
                try stdout.interface.flush();
                return;
            }

            if (!managed_mod.isInstalled(model_id, allocator)) return error.ModelNotInstalled;
            const fit = try managed_mod.verifyActiveSelectionFits(model_id, support.vram_budget_bytes, allocator);
            if (!fit.fits_current_gpu) return error.ModelDoesNotFit;
            try managed_mod.writeActiveSelection(model_id, allocator);

            var stdout_buffer: [1024]u8 = undefined;
            var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);
            try stdout.interface.print("Active model set to {s}\n", .{model_id});
            try stdout.interface.flush();
        },
        .model_rm => {
            const model_id = config.command_model_id orelse return error.MissingArgValue;
            _ = catalog_mod.find(model_id) orelse return error.UnknownManagedModel;

            if (try tryRemoveManagedModelViaLocalServer(config.port, model_id, config.command_force, allocator)) |server_response| {
                defer {
                    var owned = server_response;
                    owned.deinit(allocator);
                }

                if (server_response.status >= 200 and server_response.status < 300) {
                    try printManagedRemoveSummary(model_id, .{
                        .unloaded_from_gpu = jsonFieldIsTrue(server_response.body, "unloaded_from_gpu"),
                        .cleared_active_selection = jsonFieldIsTrue(server_response.body, "cleared_active_selection"),
                        .deleted_model = jsonFieldIsTrue(server_response.body, "deleted_model"),
                        .deleted_manifest = jsonFieldIsTrue(server_response.body, "deleted_manifest"),
                        .removed_dir = jsonFieldIsTrue(server_response.body, "removed_dir"),
                    });
                    return;
                }

                try printCommandError(extractJsonMessage(server_response.body) orelse "Managed model removal failed through the local server.");
                return error.CommandAlreadyReported;
            }

            const removed = try managed_mod.removeInstalledModel(model_id, allocator);
            const cleared_active_selection = try managed_mod.clearActiveSelectionIfMatches(model_id, allocator);
            try printManagedRemoveSummary(model_id, .{
                .unloaded_from_gpu = false,
                .cleared_active_selection = cleared_active_selection,
                .deleted_model = removed.deleted_model,
                .deleted_manifest = removed.deleted_manifest,
                .removed_dir = removed.removed_dir,
            });
        },
        .chat,
        .run => {},
    }
}

/// Build the static decode graph from GGUF metadata and write debugging artifacts.
/// Only available on Vulkan backend (loader.zig depends on Vulkan until T010-T014 refactor).
const exportDecodeGraphArtifacts = if (gpu.is_vulkan) exportDecodeGraphArtifactsImpl else (struct {
    fn f(_: []const u8, _: ?[]const u8, _: ?[]const u8, _: std.mem.Allocator) !void {
        log.warn("Graph export not yet available on Metal backend", .{});
    }
}).f;

fn runServer(
    _: anytype,
    _: *tokenizer_mod.Tokenizer,
    _: anytype,
    _: Config,
    _: std.mem.Allocator,
) !void {
    return error.ServerModeUnavailableOnThisBackend;
}

fn exportDecodeGraphArtifactsImpl(
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
        std.fs.File.stderr().writeAll(helpText(false)) catch {};
        std.process.exit(1);
    };

    if (config.show_help) {
        std.fs.File.stdout().writeAll(helpText(config.show_help_all)) catch {};
        return;
    }

    if (config.check) {
        var check_target = resolveCheckTarget(config, allocator) catch |err| {
            log.err("Failed to resolve model for diagnostics: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
        defer check_target.deinit(allocator);

        diagnostics_mod.run(.{
            .device_index = config.device_index,
            .model_path = check_target.model_path,
            .managed_model = check_target.managed_model,
            .shader_dir = if (gpu.is_metal) "src/shaders/metal" else "zig-out/share/zinc/shaders",
        }, allocator) catch |err| {
            log.err("Diagnostics completed with error: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
        return;
    }

    if (config.command != .run and config.command != .chat) {
        runModelCommand(config, allocator) catch |err| {
            if (err == error.CommandAlreadyReported) {
                std.process.exit(1);
            }
            if (err == error.GpuDetectionUnavailable) {
                std.process.exit(1);
            }
            log.err("Model command failed: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
        return;
    }

    is_debug_mode = config.debug or std.posix.getenv("ZINC_DEBUG") != null;

    const resolved_model: ?ResolvedStartupModel = blk: {
        break :blk resolveStartupModel(config, allocator) catch |err| {
            if (err == error.NoModelSpecified) {
                if (config.command == .chat) {
                    log.info("No startup model specified; starting chat server with no model loaded.", .{});
                    break :blk null;
                }
                log.warn("No model specified (-m/--model or --model-id) and no active managed model is configured. Use --help for common usage or --help-all for developer flags.", .{});
                return;
            }
            log.err("Failed to resolve model: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
    };
    defer if (resolved_model) |startup_model| {
        var owned = startup_model;
        owned.deinit(allocator);
    };

    const model_path = if (resolved_model) |model| model.spec.model_path else null;
    if (model_path) |path| {
        log.info("Model: {s}", .{path});
    }

    const wants_graph_artifacts = config.graph_report_path != null or config.graph_dot_path != null;
    if (wants_graph_artifacts and config.prompt == null) {
        if (model_path) |path| {
            exportDecodeGraphArtifacts(path, config.graph_report_path, config.graph_dot_path, allocator) catch |err| {
                log.err("Failed to export decode graph artifacts: {s}", .{@errorName(err)});
                std.process.exit(1);
            };
            return;
        }
        log.warn("Ignoring graph export flags because no startup model is loaded.", .{});
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

        log.info("ZINC Metal backend — Apple Silicon (public GPU family {s})", .{@tagName(device.chip)});
        log.info("Memory: {d} GB | Max buffer: {d} GB", .{
            device.totalMemory() / (1024 * 1024 * 1024),
            device.maxBufferSize() / (1024 * 1024 * 1024),
        });
        log.info(
            "Metal caps: apple7={any} apple8={any} apple9={any} apple10={any} mac2={any} unified={any} raytracing={any} tgmem={d} KiB working-set={d} GiB",
            .{
                device.caps.supports_apple7,
                device.caps.supports_apple8,
                device.caps.supports_apple9,
                device.caps.supports_apple10,
                device.caps.supports_mac2,
                device.caps.has_unified_memory,
                device.caps.supports_raytracing,
                device.maxThreadgroupMemoryLength() / 1024,
                device.recommendedMaxWorkingSetSize() / (1024 * 1024 * 1024),
            },
        );
        log.info(
            "Inference hints: simdgroup-width comes from pipeline threadExecutionWidth; apple10 => investigate TensorOps/M5 neural accelerators for large GEMMs; unified memory => avoid staging copies; raytracing is irrelevant for inference",
            .{},
        );

        if (config.prompt) |prompt| {
            // Load model (zero-copy mmap) for prompt-mode execution.
            var model = metal_loader.load(model_path.?, device.ctx, allocator) catch |err| {
                log.err("Failed to load model: {s}", .{@errorName(err)});
                std.process.exit(1);
            };
            defer model.deinit();

            log.info("Prompt: {s}", .{prompt});

            var tokenizer = tokenizer_mod.Tokenizer.initFromGGUF(&model.gguf_file, allocator) catch |err| {
                log.err("Failed to init tokenizer from GGUF: {s}", .{@errorName(err)});
                std.process.exit(1);
            };
            defer tokenizer.deinit();

            var prepared_prompt = try prepareCliPrompt(&tokenizer, prompt, config.chat, allocator);
            defer prepared_prompt.deinit(allocator);
            if (config.chat) {
                log.info("Prompt mode: chat template ({d} chars)", .{prepared_prompt.text.len});
            }

            const prompt_tokens = try tokenizer.encodePrompt(prepared_prompt.text, allocator);
            defer allocator.free(prompt_tokens);

            log.info("Prompt tokens ({d}): {any}", .{ prompt_tokens.len, prompt_tokens[0..@min(prompt_tokens.len, 15)] });

            // Initialize inference engine
            var engine = forward_metal.InferenceEngine.init(&model, &device, allocator, .{
                .profile_enabled = config.profile,
                .debug_validation_enabled = config.profile and config.debug,
            }) catch |err| {
                log.err("Failed to init Metal inference engine: {s}", .{@errorName(err)});
                std.process.exit(1);
            };
            defer engine.deinit();

            // Generate
            const output_tokens = forward_metal.generate(&engine, prompt_tokens, config.max_tokens, tokenizer.eosId(), allocator) catch |err| {
                log.err("Failed to generate: {s}", .{@errorName(err)});
                std.process.exit(1);
            };
            defer allocator.free(output_tokens);

            if (output_tokens.len == 0) {
                log.warn("Metal decode loop not yet implemented. Engine initialized successfully with {d} pipelines.", .{9});
            } else {
                if (config.profile) {
                    const vocab_size = model.config.vocab_size;
                    const logits_ptr: [*]const f32 = @ptrCast(@alignCast(engine.logits_buf.cpu_ptr.?));
                    const logits = logits_ptr[0..vocab_size];
                    var top_ids: [5]u32 = .{ 0, 0, 0, 0, 0 };
                    var top_vals: [5]f32 = .{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) };
                    for (logits, 0..) |v, i| {
                        if (v <= top_vals[4]) continue;
                        top_vals[4] = v;
                        top_ids[4] = @intCast(i);
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
                    for (0..5) |k| {
                        var dec_buf: [256]u8 = undefined;
                        const tok_str = tokenizer.decodeToken(top_ids[k], &dec_buf);
                        log.info("  metal prefill logit #{d}: id={d} val={d:.4} \"{s}\"", .{ k, top_ids[k], top_vals[k], tok_str });
                    }
                }

                // Decode tokens to text
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
                log.info("Output ({d} tokens): {s}", .{ output_tokens.len, output_text });
            }
        } else {
            log.info("Server mode — port {d}, max {d} concurrent requests", .{ config.port, config.max_parallel });

            var manager = if (resolved_model) |startup_model|
                model_manager_mod.ModelManager.init(startup_model.spec, &device, allocator) catch |err| {
                    log.err("Failed to init Metal model manager: {s}", .{@errorName(err)});
                    std.process.exit(1);
                }
            else
                model_manager_mod.ModelManager.initEmpty(&device, allocator);
            defer manager.deinit();

            runHttpServer(config, &manager, allocator);
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

    // Determine shader directory
    const shader_dir = "zig-out/share/zinc/shaders";

    if (config.prompt) |prompt| {
        log.debug("Prompt: {s}", .{prompt});

        var cmd_pool = try CommandPool.init(&vk_instance);
        defer cmd_pool.deinit();

        var model = loader_mod.load(model_path.?, &vk_instance, &cmd_pool, allocator) catch |err| {
            log.err("Failed to load model: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
        defer model.deinit(&vk_instance);

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

        if (config.profile) {
            engine.enableProfiling() catch |err| {
                log.warn("Failed to enable profiling: {s}", .{@errorName(err)});
            };
        }
        if (config.debug) {
            engine.enableLogitsReadback();
        }

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

        // Tokenize prompt into caller-owned storage. This keeps CLI and server
        // prompt construction on the same code path, including BOS handling.
        const prompt_tokens = try tokenizer.encodePrompt(prepared_prompt.text, allocator);
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
        const output_tokens = try forward_mod.generate(&engine, prompt_tokens, config.max_tokens, tokenizer.eosId(), allocator);
        defer allocator.free(output_tokens);

        // Output token IDs
        log.debug("Output tokens ({d}): {any}", .{
            output_tokens.len,
            output_tokens[0..@min(output_tokens.len, 20)],
        });

        // Debug: dump first 5 generated tokens with their vocabulary text
        if (config.debug) {
            const show_n = @min(output_tokens.len, 5);
            for (0..show_n) |ti| {
                const tok_str = if (output_tokens[ti] < tokenizer.vocab.len) tokenizer.vocab[output_tokens[ti]] else "?";
                log.debug("  gen[{d}]: id={d} \"{s}\"", .{ ti, output_tokens[ti], tok_str });
            }
        }
        // Check specific token logits (Paris=11751, not=524)
        if (config.debug) {
            const logits_ptr2: [*]const f32 = @ptrCast(@alignCast(engine.logits_staging.mapped.?));
            log.debug("  logit[11751 'Paris']={d:.4} logit[524 'not']={d:.4} logit[264 'a']={d:.4}", .{
                logits_ptr2[11751], logits_ptr2[524], logits_ptr2[264],
            });
        }
        // Debug: dump top-5 logits from the last decode step
        if (config.debug) {
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

        var manager = if (resolved_model) |startup_model|
            model_manager_mod.ModelManager.init(startup_model.spec, &vk_instance, gpu_config, shader_dir, allocator) catch |err| {
                log.err("Failed to init model manager: {s}", .{@errorName(err)});
                std.process.exit(1);
            }
        else
            model_manager_mod.ModelManager.initEmpty(&vk_instance, gpu_config, shader_dir, allocator);
        defer manager.deinit();
        runHttpServer(config, &manager, allocator);
    }
}

test "parseArgs: defaults" {
    const args = [_][:0]const u8{"zinc"};
    const config = try parseArgs(&args);
    try std.testing.expectEqual(@as(u16, 8080), config.port);
    try std.testing.expectEqual(@as(u32, 4096), config.context_length);
    try std.testing.expectEqual(@as(u8, 0), config.kv_quant);
    try std.testing.expect(config.model_path == null);
    try std.testing.expect(config.model_id == null);
    try std.testing.expect(config.prompt == null);
    try std.testing.expect(!config.chat);
    try std.testing.expectEqual(Command.run, config.command);
}

test "parseArgs: full args" {
    const args = [_][:0]const u8{
        "zinc",           "-m",         "model.gguf",  "--model-id", "qwen35-2b-q4k-m",
        "-p",             "9090",       "-d",          "1",          "-c",
        "8192",           "--parallel", "8",           "--prompt",   "hello",
        "--max-tokens",   "32",         "--chat",      "--kv-quant", "3",
        "--graph-report", "graph.json", "--graph-dot", "graph.dot",
    };
    const config = try parseArgs(&args);
    try std.testing.expectEqualStrings("model.gguf", config.model_path.?);
    try std.testing.expectEqualStrings("qwen35-2b-q4k-m", config.model_id.?);
    try std.testing.expectEqual(@as(u16, 9090), config.port);
    try std.testing.expectEqual(@as(u32, 1), config.device_index);
    try std.testing.expectEqual(@as(u32, 8192), config.context_length);
    try std.testing.expectEqual(@as(u32, 8), config.max_parallel);
    try std.testing.expectEqualStrings("hello", config.prompt.?);
    try std.testing.expectEqual(@as(u32, 32), config.max_tokens);
    try std.testing.expect(config.chat);
    try std.testing.expectEqual(@as(u8, 3), config.kv_quant);
    try std.testing.expectEqualStrings("graph.json", config.graph_report_path.?);
    try std.testing.expectEqualStrings("graph.dot", config.graph_dot_path.?);
}

test "parseArgs: help flag" {
    const args = [_][:0]const u8{ "zinc", "--help" };
    const config = try parseArgs(&args);
    try std.testing.expect(config.show_help);
    try std.testing.expect(!config.show_help_all);
}

test "parseArgs: help-all flag" {
    const args = [_][:0]const u8{ "zinc", "--help-all" };
    const config = try parseArgs(&args);
    try std.testing.expect(config.show_help);
    try std.testing.expect(config.show_help_all);
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

test "parseArgs: max tokens flag" {
    const args = [_][:0]const u8{ "zinc", "--prompt", "hi", "-n", "12" };
    const config = try parseArgs(&args);
    try std.testing.expectEqual(@as(u32, 12), config.max_tokens);
}

test "parseArgs: chat flag" {
    const args = [_][:0]const u8{ "zinc", "--prompt", "hi", "--chat" };
    const config = try parseArgs(&args);
    try std.testing.expect(config.chat);
    try std.testing.expectEqualStrings("hi", config.prompt.?);
}

test "helpText: short help hides developer-only flags" {
    const text = helpText(false);
    try std.testing.expect(std.mem.indexOf(u8, text, "--help-all") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "Common options:") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "--graph-report") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "--profile") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "--debug") == null);
}

test "helpText: full help includes developer-only flags" {
    const text = helpText(true);
    try std.testing.expect(std.mem.indexOf(u8, text, "Analysis and developer options:") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "--graph-report") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "--profile") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "--debug") != null);
}

test "parseArgs: managed model subcommands" {
    const list_args = [_][:0]const u8{ "zinc", "model", "list", "--all" };
    const list_config = try parseArgs(&list_args);
    try std.testing.expectEqual(Command.model_list, list_config.command);
    try std.testing.expect(list_config.show_all_models);

    const pull_args = [_][:0]const u8{ "zinc", "model", "pull", "qwen35-2b-q4k-m" };
    const pull_config = try parseArgs(&pull_args);
    try std.testing.expectEqual(Command.model_pull, pull_config.command);
    try std.testing.expectEqualStrings("qwen35-2b-q4k-m", pull_config.command_model_id.?);

    const active_args = [_][:0]const u8{ "zinc", "model", "active" };
    const active_config = try parseArgs(&active_args);
    try std.testing.expectEqual(Command.model_active, active_config.command);

    const rm_args = [_][:0]const u8{ "zinc", "model", "rm", "-f", "qwen35-2b-q4k-m" };
    const rm_config = try parseArgs(&rm_args);
    try std.testing.expectEqual(Command.model_rm, rm_config.command);
    try std.testing.expect(rm_config.command_force);
    try std.testing.expectEqualStrings("qwen35-2b-q4k-m", rm_config.command_model_id.?);
}

test "parseArgs: chat command" {
    const args = [_][:0]const u8{ "zinc", "chat", "--model-id", "qwen35-2b-q4k-m" };
    const config = try parseArgs(&args);
    try std.testing.expectEqual(Command.chat, config.command);
    try std.testing.expectEqualStrings("qwen35-2b-q4k-m", config.model_id.?);
    try std.testing.expectEqual(@as(u16, 9090), config.port);
}

test "parseArgs: chat command preserves explicit port before subcommand" {
    const args = [_][:0]const u8{ "zinc", "-p", "8088", "chat" };
    const config = try parseArgs(&args);
    try std.testing.expectEqual(Command.chat, config.command);
    try std.testing.expectEqual(@as(u16, 8088), config.port);
}

test "parseArgs: chat command preserves explicit port after subcommand" {
    const args = [_][:0]const u8{ "zinc", "chat", "-p", "8088" };
    const config = try parseArgs(&args);
    try std.testing.expectEqual(Command.chat, config.command);
    try std.testing.expectEqual(@as(u16, 8088), config.port);
}

test "parseArgs: chat command rejects prompt mode" {
    const args = [_][:0]const u8{ "zinc", "chat", "--prompt", "hello" };
    try std.testing.expectError(error.ChatCommandDoesNotTakePrompt, parseArgs(&args));
}

test "parseArgs: chat command rejects model subcommands" {
    const args = [_][:0]const u8{ "zinc", "chat", "model", "list" };
    try std.testing.expectError(error.UnknownArgument, parseArgs(&args));
}

test "resolveCheckTarget returns general diagnostics target when no model is specified" {
    const config = Config{};
    var target = try resolveCheckTarget(config, std.testing.allocator);
    defer target.deinit(std.testing.allocator);

    try std.testing.expect(target.model_path == null);
    try std.testing.expect(target.managed_model == null);
}

test "resolveCheckTarget uses raw gguf path when no managed id is provided" {
    const config = Config{ .model_path = "model.gguf" };
    var target = try resolveCheckTarget(config, std.testing.allocator);
    defer target.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("model.gguf", target.model_path.?);
    try std.testing.expect(target.managed_model == null);
}

test "resolveCheckTarget prefers managed model id over raw gguf path" {
    const config = Config{
        .model_id = "qwen35-2b-q4k-m",
        .model_path = "raw.gguf",
    };
    var target = try resolveCheckTarget(config, std.testing.allocator);
    defer target.deinit(std.testing.allocator);

    try std.testing.expect(target.managed_model != null);
    try std.testing.expectEqualStrings("qwen35-2b-q4k-m", target.managed_model.?.id);
    if (target.model_path) |path| {
        try std.testing.expect(!std.mem.eql(u8, path, "raw.gguf"));
    }
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
