//! CLI entrypoints for configuring ZINC and starting local inference.
//! @section CLI & Entrypoints
//! This module wires together GPU initialization, model loading, tokenization,
//! and the single-process decode loop used for prompt-mode execution.
const builtin = @import("builtin");
const std = @import("std");
const build_info = @import("build_info.zig");
const gpu = @import("gpu/interface.zig");
const runtime_assets = @import("runtime_assets.zig");
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
const hf_mod = @import("model/hf.zig");
const tokenizer_mod = @import("model/tokenizer.zig");
const graph_mod = @import("compute/graph.zig");
const memory_plan = @import("gpu/memory_plan.zig");
const process_lock_mod = @import("gpu/process_lock.zig");
const server_runtime = @import("server/runtime.zig");
// These modules import vulkan/ transitively — only available on Linux until T010-T014 refactor.
// On macOS they are stubbed out; the GPU abstraction refactor will make them platform-independent.
const loader_mod = if (gpu.is_vulkan) @import("model/loader.zig") else struct {};
const architecture_mod = if (gpu.is_vulkan) @import("model/architecture.zig") else struct {};
const forward_mod = if (gpu.is_vulkan) @import("compute/vulkan_inference.zig") else struct {};

// Backend-specific imports (only one branch compiles per platform)
const instance_mod = if (gpu.is_vulkan) @import("vulkan/instance.zig") else gpu.backend;
const gpu_detect = if (gpu.is_vulkan) @import("vulkan/gpu_detect.zig") else struct {};
// CUDA: backend modules (only resolve under -Dbackend=cuda; stubbed elsewhere)
const cuda_device_mod = if (gpu.is_cuda) @import("cuda/device.zig") else struct {};
const loader_cuda_mod = if (gpu.is_cuda) @import("model/loader_cuda.zig") else struct {};
const forward_cuda_mod = if (gpu.is_cuda) @import("compute/forward_cuda.zig") else struct {};
const forward_cuda_gemma_mod = if (gpu.is_cuda) @import("compute/forward_cuda_gemma.zig") else struct {};
const forward_inkling_mod = if (gpu.is_rocm) @import("compute/forward_inkling.zig") else struct {};
// Effort 28 (inc 3 / 3b): CUDA multi-tenant serving engine (gemma dense). Only
// referenced on the CUDA backend; the struct{} stub keeps other backends clean.
const cuda_serve_mod = if (gpu.is_cuda) @import("server/cuda_serve.zig") else struct {};
const accelerator_catalog = @import("server/accelerator_model_catalog.zig");
const model_replacement = @import("server/model_replacement.zig");
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
const inference_timing_log = std.log.scoped(.forward);
const accelerator_name = if (gpu.is_rocm) "ROCm" else "CUDA";

/// Global flag enabling verbose debug log output when `--debug` is passed or the `ZINC_DEBUG` env var is set.
pub var is_debug_mode: bool = false;

/// Zig standard library options — sets log level to debug and wires the custom log function.
pub const std_options = std.Options{
    .log_level = .debug,
    .logFn = myLogFn,
};

/// Custom log handler that filters debug messages unless `is_debug_mode` is set.
/// @param level Severity level; debug messages are suppressed when `is_debug_mode` is false.
/// @param scope Call-site scope tag; displayed as `(<scope>):` or `: ` for the default scope.
/// @param format Comptime format string passed through to `std.debug.print`.
/// @param args Runtime arguments matched to `format`.
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
    if (gpu.is_cuda) {
        // CUDA: force-compile the CUDA backend modules + the shared catalog/
        // managed-model code the CUDA CLI path reuses.
        _ = @import("cuda/device.zig");
        _ = @import("cuda/buffer.zig");
        _ = @import("cuda/pipeline.zig");
        _ = @import("cuda/command.zig");
        _ = @import("model/loader_cuda.zig");
        _ = @import("compute/forward_cuda.zig");
        _ = @import("compute/forward_cuda_gemma.zig");
        _ = @import("model/catalog.zig");
        _ = @import("model/managed.zig");
    }
    // Platform-independent modules
    _ = @import("model/config.zig");
    _ = @import("model/inkling.zig");
    _ = @import("model/hf.zig");
    _ = @import("model/gguf.zig");
    _ = @import("model/tokenizer.zig");
    _ = @import("compute/graph.zig");
    _ = @import("build_info.zig");
    _ = @import("runtime_assets.zig");
    _ = @import("regression_tests.zig");
    _ = @import("server/http.zig");
    _ = @import("server/runtime.zig");
    if (builtin.is_test) {
        _ = @import("server/mtp_policy_test.zig");
        _ = accelerator_catalog;
        _ = model_replacement;
    }
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
    /// Hugging Face `owner/repo[:quant]` spec to download and load (`-hf`).
    hf_spec: ?[]const u8 = null,
    /// HTTP server port.
    port: u16 = 8080,
    /// GPU device index used when explicitly set with `-d/--device`.
    device_index: u32 = 0,
    /// True when `device_index` came from the CLI instead of the backend default.
    device_index_explicit: bool = false,
    /// Max sequence length. `null` means auto-size from GPU memory at load.
    context_length: ?u32 = null,
    /// Max concurrent requests.
    max_parallel: u32 = 4,
    /// CLI prompt text.
    prompt: ?[]const u8 = null,
    /// Optional system turn prepended when an explicit CLI chat prompt is used.
    system_prompt: ?[]const u8 = null,
    /// Maximum CLI decode tokens.
    max_tokens: u32 = 256,
    /// Wrap CLI prompt in the model's chat template before tokenization.
    chat: bool = false,
    /// Keep CLI prompt as a raw completion even for chat-first templates.
    raw_prompt: bool = false,
    /// KV-cache quantization bit-width: 0 disables TurboQuant, 2/3/4 select the quant level.
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
    /// Print version/build metadata and exit.
    show_version: bool = false,
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
    /// Output `model list` results as pretty-printed JSON.
    json_output: bool = false,

    fn gpuDevicePreference(self: Config) u32 {
        if (comptime gpu.is_vulkan) {
            return if (self.device_index_explicit) self.device_index else instance_mod.auto_select_device_index;
        }
        return self.device_index;
    }
};

/// Top-level CLI subcommands parsed from argv.
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

fn openBrowser(port: u16) void {
    var url_buf: [64]u8 = undefined;
    const url = std.fmt.bufPrint(&url_buf, "http://localhost:{d}", .{port}) catch return;
    const opener = if (comptime @import("builtin").os.tag == .macos) "open" else "xdg-open";
    var child = std.process.Child.init(&.{ opener, url }, std.heap.page_allocator);
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Ignore;
    child.stderr_behavior = .Ignore;
    _ = child.spawnAndWait() catch {};
}

fn reportGpuProcessLockError(err: anyerror, backend: process_lock_mod.Backend, device_index: u32) noreturn {
    switch (err) {
        error.GpuAlreadyReserved => log.err(
            "GPU {s}:{d} is already reserved by another zinc process. Stop the other instance before loading a second model on the same GPU.",
            .{ @tagName(backend), device_index },
        ),
        else => log.err("Failed to acquire GPU process lock for {s}:{d}: {s}", .{
            @tagName(backend),
            device_index,
            @errorName(err),
        }),
    }
    std.process.exit(1);
}

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

    if (config.command == .chat) {
        openBrowser(config.port);
    }

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
    \\ZINC — Zig INferenCe Engine for consumer GPUs and Apple Silicon
    \\
    \\Usage:
    \\  zinc -m <model.gguf> --prompt "Hello"
    \\  zinc -m <model.gguf> [-p 8080]
    \\  zinc chat [-m <model.gguf> | --model-id <id>] [-p 9090]
    \\  zinc --model-id <id> [--prompt "Hello"]
    \\  zinc -hf <owner/model[:quant]> [--prompt "Hello"]
    \\  zinc --check [-m <model.gguf> | --model-id <id>]
    \\  zinc model <list|pull|use|active|rm> [args]
    \\
    \\Common options:
    \\  -m, --model <path>       GGUF model file to load
    \\  --model-id <id>          Managed model id from the local catalog/cache
    \\  -hf, --hf-repo <spec>    Hugging Face repo (owner/model[:quant]) to download and load
    \\  --prompt <text>          Run one prompt in CLI mode instead of starting the server
    \\  --chat                   Apply the model chat template to --prompt
    \\  --system-prompt <text>   Add an explicit system turn (requires --chat)
    \\  --raw                    Do not auto-apply chat templates to --prompt
    \\  -n, --max-tokens <n>     Max generated tokens in CLI mode (default: 256)
    \\  -d, --device <id>        GPU device index (default: auto, prefers discrete Vulkan GPU)
    \\  -c, --context <size>     Context length (default: auto — sized to GPU memory; pass 0 to force auto)
    \\  --kv-quant <bits>        TurboQuant KV cache bits: 0/2/3/4 (default: 0)
    \\
    \\Server options:
    \\  -p, --port <port>        Server port (default: 8080)
    \\  --parallel <n>           Max concurrent requests (default: 4)
    \\  chat                     Start the server on port 9090 and open the built-in chat UI in your browser
    \\
    \\Model management:
    \\  model list [--all] [--json] List managed models (--json for machine-readable output)
    \\  model pull <id>          Download a supported managed model into the local cache
    \\  model use <id>           Set the active managed model for future runs
    \\  model active             Print the active managed model
    \\  model rm [-f] <id>       Remove a cached model (catalog or hf-- id); -f unloads it first if active
    \\
    \\Diagnostics:
    \\  --check                  Run system diagnostics and verify dependencies
    \\  --version                Show version and build metadata
    \\  -h, --help               Show this help
    \\  --help-all               Show diagnostics and developer-only flags too
    \\
    \\Use `--help-all` to show graph export, profiling, and debug flags.
    \\
;

const banner_full =
    \\ZINC — Zig INferenCe Engine for consumer GPUs and Apple Silicon
    \\
    \\Usage:
    \\  zinc -m <model.gguf> --prompt "Hello"
    \\  zinc -m <model.gguf> [-p 8080]
    \\  zinc chat [-m <model.gguf> | --model-id <id>] [-p 9090]
    \\  zinc --model-id <id> [--prompt "Hello"]
    \\  zinc -hf <owner/model[:quant]> [--prompt "Hello"]
    \\  zinc --check [-m <model.gguf> | --model-id <id>]
    \\  zinc model <list|pull|use|active|rm> [args]
    \\
    \\Common options:
    \\  -m, --model <path>       GGUF model file to load
    \\  --model-id <id>          Managed model id from the local catalog/cache
    \\  -hf, --hf-repo <spec>    Hugging Face repo (owner/model[:quant]) to download and load
    \\  --prompt <text>          Run one prompt in CLI mode instead of starting the server
    \\  --chat                   Apply the model chat template to --prompt
    \\  --system-prompt <text>   Add an explicit system turn (requires --chat)
    \\  --raw                    Do not auto-apply chat templates to --prompt
    \\  -n, --max-tokens <n>     Max generated tokens in CLI mode (default: 256)
    \\  -d, --device <id>        GPU device index (default: auto, prefers discrete Vulkan GPU)
    \\  -c, --context <size>     Context length (default: auto — sized to GPU memory; pass 0 to force auto)
    \\  --kv-quant <bits>        TurboQuant KV cache bits: 0/2/3/4 (default: 0)
    \\
    \\Server options:
    \\  -p, --port <port>        Server port (default: 8080)
    \\  --parallel <n>           Max concurrent requests (default: 4)
    \\  chat                     Start the server on port 9090 and open the built-in chat UI in your browser
    \\
    \\Model management:
    \\  model list [--all] [--json] List managed models (--json for machine-readable output)
    \\  model pull <id>          Download a supported managed model into the local cache
    \\  model use <id>           Set the active managed model for future runs
    \\  model active             Print the active managed model
    \\  model rm [-f] <id>       Remove a cached model (catalog or hf-- id); -f unloads it first if active
    \\
    \\Diagnostics:
    \\  --check                  Run system diagnostics and verify dependencies
    \\  --version                Show version and build metadata
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
        } else if (std.mem.eql(u8, arg, "--version")) {
            config.show_version = true;
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
        } else if (std.mem.eql(u8, arg, "-hf") or std.mem.eql(u8, arg, "--hf-repo")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.hf_spec = args[i];
        } else if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--port")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.port = std.fmt.parseInt(u16, args[i], 10) catch return error.InvalidPort;
            port_explicit = true;
        } else if (std.mem.eql(u8, arg, "-d") or std.mem.eql(u8, arg, "--device")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.device_index = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidDevice;
            config.device_index_explicit = true;
        } else if (std.mem.eql(u8, arg, "-c") or std.mem.eql(u8, arg, "--context")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            const parsed_ctx = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidContext;
            config.context_length = if (parsed_ctx == 0) null else parsed_ctx;
        } else if (std.mem.eql(u8, arg, "--parallel")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.max_parallel = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidParallel;
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.prompt = args[i];
        } else if (std.mem.eql(u8, arg, "--system-prompt")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.system_prompt = args[i];
        } else if (std.mem.eql(u8, arg, "-n") or std.mem.eql(u8, arg, "--max-tokens")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.max_tokens = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidMaxTokens;
        } else if (std.mem.eql(u8, arg, "--chat")) {
            config.chat = true;
        } else if (std.mem.eql(u8, arg, "--raw")) {
            config.raw_prompt = true;
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
        } else if (std.mem.eql(u8, arg, "--json")) {
            config.json_output = true;
        } else if (std.mem.eql(u8, arg, "-f") or std.mem.eql(u8, arg, "--force")) {
            config.command_force = true;
        } else {
            return error.UnknownArgument;
        }
    }

    if (config.command == .chat and config.prompt != null) {
        return error.ChatCommandDoesNotTakePrompt;
    }
    if (config.hf_spec != null and (config.model_path != null or config.model_id != null)) {
        return error.ConflictingModelSources;
    }
    if (config.chat and config.raw_prompt) {
        return error.ConflictingPromptModes;
    }
    if (config.system_prompt != null and (!config.chat or config.prompt == null)) {
        return error.SystemPromptRequiresChat;
    }

    return config;
}

fn shouldAutoChatCliPrompt(tokenizer: *const tokenizer_mod.Tokenizer, prompt: []const u8) bool {
    const tmpl = tokenizer.chat_template orelse return false;
    if (tokenizer.detectTemplateKind() == .inkling2) {
        return std.mem.indexOf(u8, prompt, "<|message_user|>") == null;
    }
    // ATEM (Muse Glimmer): a reasoning-tuned model that degenerates badly on
    // raw completion — auto-apply its chat template unless the caller already
    // wrote the turn markers themselves.
    if (std.mem.indexOf(u8, tmpl, "<atem:") != null) {
        return std.mem.indexOf(u8, prompt, "<|start|>") == null;
    }
    if (std.mem.indexOf(u8, tmpl, "<|turn>") == null) return false;
    if (std.mem.indexOf(u8, prompt, "<|turn>") != null) return false;
    return true;
}

/// Engine-level stop id for CLI generation: ATEM chats end their turn with
/// `<|eot|>` (distinct from EOS), so templated Muse prompts stop there.
fn cliGenerationStopId(tokenizer: *const tokenizer_mod.Tokenizer, use_chat_prompt: bool) u32 {
    if (use_chat_prompt and tokenizer.detectTemplateKind() == .atem) {
        if (tokenizer.eot_id) |eot| return eot;
    }
    return tokenizer.eosId();
}

fn indexOfAsciiIgnoreCase(haystack: []const u8, needle: []const u8) ?usize {
    if (needle.len == 0) return 0;
    if (needle.len > haystack.len) return null;

    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        var j: usize = 0;
        while (j < needle.len and std.ascii.toLower(haystack[i + j]) == std.ascii.toLower(needle[j])) : (j += 1) {}
        if (j == needle.len) return i;
    }
    return null;
}

fn isIncludedWordByte(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_' or c == '-';
}

fn requestedIncludedWord(prompt: []const u8) ?[]const u8 {
    const marker = "include the word ";
    const marker_idx = indexOfAsciiIgnoreCase(prompt, marker) orelse return null;
    var start = marker_idx + marker.len;
    while (start < prompt.len and
        (prompt[start] == ' ' or prompt[start] == '\t' or prompt[start] == '"' or prompt[start] == '\'' or prompt[start] == '`')) : (start += 1)
    {}

    var end = start;
    while (end < prompt.len and isIncludedWordByte(prompt[end])) : (end += 1) {}
    if (end == start) return null;
    return prompt[start..end];
}

fn promptFingerprint(text: []const u8) u64 {
    return std.hash.Wyhash.hash(0, text);
}

fn prepareCliPrompt(
    tokenizer: *const tokenizer_mod.Tokenizer,
    prompt: []const u8,
    chat: bool,
    system_prompt: ?[]const u8,
    allocator: std.mem.Allocator,
) !PreparedPrompt {
    if (!chat) {
        return .{ .text = prompt };
    }

    // Llama 3 instruct models need a system prompt for best results.
    // Detect Llama-style templates (start_header_id) and prepend system message.
    const template_needs_system = if (tokenizer.chat_template) |tmpl|
        std.mem.indexOf(u8, tmpl, "start_header_id") != null
    else
        false;

    if (system_prompt != null or template_needs_system) {
        const roles = [_][]const u8{ "system", "user" };
        const system_content = system_prompt orelse "You are a helpful assistant.";
        const contents = [_][]const u8{ system_content, prompt };
        const chat_capacity = prompt.len + system_content.len + 512;
        const chat_buf = try allocator.alloc(u8, chat_capacity);
        errdefer allocator.free(chat_buf);
        const formatted = try tokenizer.applyChatTemplate(&roles, &contents, chat_buf);
        return .{ .text = formatted, .owned_buf = chat_buf };
    }

    const use_gemma_turn_template = if (tokenizer.chat_template) |tmpl|
        std.mem.indexOf(u8, tmpl, "<|turn>") != null
    else
        false;

    var strengthened_prompt_buf: ?[]u8 = null;
    defer if (strengthened_prompt_buf) |buf| allocator.free(buf);

    const prompt_for_template = if (use_gemma_turn_template) blk: {
        const word = requestedIncludedWord(prompt) orelse break :blk prompt;
        const strengthened = try std.fmt.allocPrint(
            allocator,
            "Begin the answer with the exact word {s}. That word must be the first output word, then continue with the requested answer.\n\n{s}",
            .{ word, prompt },
        );
        strengthened_prompt_buf = strengthened;
        break :blk strengthened;
    } else prompt;

    const roles = [_][]const u8{"user"};
    const contents = [_][]const u8{prompt_for_template};
    const chat_capacity = prompt_for_template.len + 256;
    const chat_buf = try allocator.alloc(u8, chat_capacity);
    errdefer allocator.free(chat_buf);

    const formatted = if (use_gemma_turn_template)
        // Gemma templates already define the default generation scaffold.
        // Respect that instead of forcing the explicit thinking branch.
        try tokenizer.applyChatTemplate(&roles, &contents, chat_buf)
    else if (tokenizer.supportsThinkingToggle())
        // Qwen-style ChatML models answer much more directly when the chat
        // template emits an explicit closed think block for generation.
        try tokenizer.applyChatTemplate(&roles, &contents, chat_buf)
    else
        try tokenizer.applyChatTemplate(&roles, &contents, chat_buf);
    return .{
        .text = formatted,
        .owned_buf = chat_buf,
    };
}

const cli_harmony_final_prefix = "<|channel|>final<|message|>";
const cli_harmony_stop_strs = [_][]const u8{
    "<|end|>",
    "<|return|>",
    "<|start|>",
    "<|channel|>",
};
const cli_chat_stop_strs = [_][]const u8{
    "<|im_end|>",
    "<turn|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "<|return|>",
};

fn findFirstCliStop(text: []const u8, needles: []const []const u8) ?usize {
    var first: ?usize = null;
    for (needles) |needle| {
        if (std.mem.indexOf(u8, text, needle)) |idx| {
            if (first == null or idx < first.?) first = idx;
        }
    }
    return first;
}

fn trimCliOutputText(text: []const u8, chat: bool) []const u8 {
    if (!chat) return text;
    if (std.mem.indexOf(u8, text, cli_harmony_final_prefix)) |final_start| {
        const body = text[final_start + cli_harmony_final_prefix.len ..];
        const stop_pos = findFirstCliStop(body, cli_harmony_stop_strs[0..]) orelse body.len;
        return std.mem.trim(u8, body[0..stop_pos], " \t\r\n");
    }
    if (findFirstCliStop(text, cli_chat_stop_strs[0..])) |stop_pos| {
        return std.mem.trimRight(u8, text[0..stop_pos], " \t\r\n");
    }
    return text;
}

fn resolveStartupModel(config: Config, allocator: std.mem.Allocator) !ResolvedStartupModel {
    if (config.hf_spec) |spec| {
        var stdout_buffer: [512]u8 = undefined;
        var stdout_writer = std.fs.File.stdout().writerStreaming(&stdout_buffer);
        const path = try hf_mod.ensureModel(spec, allocator, &stdout_writer.interface);
        return .{
            .spec = .{
                .model_path = path,
                .requested_context_length = config.context_length,
            },
            .owned_path = path,
        };
    }

    if (config.model_id) |model_id| {
        const path = try managed_mod.resolveInstalledModelPath(model_id, allocator);
        const model_id_copy = try allocator.dupe(u8, model_id);
        return .{
            .spec = .{
                .model_path = path,
                .managed_id = model_id_copy,
                .requested_context_length = config.context_length,
            },
            .owned_path = path,
            .owned_managed_id = model_id_copy,
        };
    }

    if (config.model_path) |model_path| {
        return .{ .spec = .{
            .model_path = model_path,
            .requested_context_length = config.context_length,
        } };
    }

    const active = try managed_mod.readActiveSelection(allocator);
    if (active) |selection| {
        const path = try managed_mod.resolveInstalledModelPath(selection.model_id, allocator);
        return .{
            .spec = .{
                .model_path = path,
                .managed_id = selection.model_id,
                .requested_context_length = config.context_length,
            },
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

    fn rocm16GbContext(self: *const ManagedGpuSupport, entry: catalog_mod.CatalogEntry) ?u32 {
        return if (gpu.is_rocm) catalog_mod.rocm16GbContext(entry, self.profile) else null;
    }

    fn describeFit(self: *const ManagedGpuSupport, entry: catalog_mod.CatalogEntry, allocator: std.mem.Allocator) managed_mod.ModelFit {
        const budget = self.vram_budget_bytes;
        if (self.rocm16GbContext(entry) != null) {
            const required = catalog_mod.rocm_16gb_min_vram_bytes;
            return .{
                .required_vram_bytes = required,
                .required_vram_with_offload_bytes = required,
                .fits_current_gpu = required <= budget,
                .fit_state = if (required <= budget) .fits else .does_not_fit,
                .exact = false,
            };
        }
        return managed_mod.describeFit(entry, budget, allocator) catch .{
            .required_vram_bytes = entry.required_vram_bytes,
            .required_vram_with_offload_bytes = catalog_mod.requiredVramWithOffload(entry),
            .fits_current_gpu = catalog_mod.fitsGpu(entry, budget),
            .fit_state = catalog_mod.fitState(entry, budget),
            .exact = false,
        };
    }
};

fn resolveManagedGpuSupport(device_index: u32, allocator: std.mem.Allocator) !ManagedGpuSupport {
    if (comptime gpu.is_rocm) {
        // The Vulkan device-index cache does not identify HIP's selected device.
        var device = try cuda_device_mod.CudaDevice.initBest(allocator);
        defer device.deinit();
        var arch: [64]u8 = undefined;
        const bytes = device.totalMemory();
        return .{
            .profile = try allocator.dupe(u8, catalog_mod.profileForRocm(device.arch(&arch), bytes)),
            .vram_budget_bytes = bytes,
            .from_cache = false,
        };
    }
    const auto_vulkan = if (comptime gpu.is_vulkan)
        device_index == instance_mod.auto_select_device_index
    else
        false;
    if (!auto_vulkan) {
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
    }

    if (gpu.is_vulkan) {
        var vk_instance = try instance_mod.Instance.init(allocator, device_index);
        defer vk_instance.deinit();

        const selected_device_index = vk_instance.selected_device_index;
        if (device_index == instance_mod.auto_select_device_index) {
            if (try managed_mod.readCachedGpuProfile(selected_device_index, allocator)) |cached| {
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
        }

        const gpu_config = gpu_detect.detect(&vk_instance);
        const profile = catalog_mod.profileForGpu(gpu_config);
        const vram_budget_bytes = vk_instance.vramBytes();

        try managed_mod.writeCachedGpuProfile(selected_device_index, profile, gpu_config.nameSlice(), vram_budget_bytes, allocator);

        return .{
            .profile = try allocator.dupe(u8, profile),
            .vram_budget_bytes = vram_budget_bytes,
            .from_cache = false,
        };
    }

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

    if (gpu.is_cuda) {
        // CUDA: probe the best NVIDIA device, derive the catalog profile from
        // the device name, and use free VRAM as the fit budget (mirrors the
        // Metal branch's working-set budget). device_index is ignored here —
        // initBest picks the highest-CC device (5090 over 4090).
        var device = try cuda_device_mod.CudaDevice.initBest(allocator);
        defer device.deinit();

        const profile = catalog_mod.profileForCuda();
        const vram_budget_bytes = blk: {
            const free = device.freeMemory();
            break :blk if (free > 0) free else device.totalMemory();
        };
        var name_buf: [256]u8 = undefined;
        const device_name = device.name(&name_buf);

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

    const support = resolveManagedGpuSupport(config.gpuDevicePreference(), allocator) catch |err| {
        if (!config.show_all_models and !config.json_output) {
            var stderr_buffer: [1024]u8 = undefined;
            var stderr = std.fs.File.stderr().writerStreaming(&stderr_buffer);
            try stderr.interface.print("Unable to initialize {s} for GPU detection: {s}\n", .{ backend_name, @errorName(err) });
            try stderr.interface.writeAll("Use `zinc model list --all` to inspect the catalog without live fit checks.\n");
            try stderr.interface.flush();
            return error.GpuDetectionUnavailable;
        }

        if (config.json_output) {
            try printManagedModelListJson(&stdout.interface, active_model_id, null, allocator);
            try stdout.interface.flush();
            return;
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

    if (config.json_output) {
        try printManagedModelListJson(&stdout.interface, active_model_id, &support, allocator);
        try stdout.interface.flush();
        return;
    }

    try stdout.interface.print(
        "Detected GPU profile: {s}{s}\n\n",
        .{
            support.profile,
            if (support.from_cache) " (cached)" else "",
        },
    );
    try stdout.interface.writeAll("ID                             Released     Status              Fit       Installed   Active   Notes\n");

    var rendered_any = false;
    var any_requires_offload = false;
    for (catalog_mod.entries) |entry| {
        const low_vram_ctx = support.rocm16GbContext(entry);
        const tested_profile_match = low_vram_ctx != null or catalog_mod.supportsProfile(entry, support.profile);
        const installed = managed_mod.isInstalled(entry.id, allocator);
        const fit = support.describeFit(entry, allocator);
        const supported_now = tested_profile_match and fit.fits_current_gpu;
        const supported_with_offload = tested_profile_match and fit.fit_state == .fits_with_offload;
        const visible = supported_now or supported_with_offload;
        if (!config.show_all_models and !visible) continue;

        rendered_any = true;
        if (supported_with_offload) any_requires_offload = true;
        const is_active = active_model_id != null and std.mem.eql(u8, active_model_id.?, entry.id);
        const status_label = if (supported_now)
            @tagName(entry.status)
        else if (supported_with_offload)
            switch (entry.status) {
                .supported => "supported (offload)",
                .experimental => "experimental (offload)",
                .hidden => "hidden (offload)",
                .deprecated => "deprecated (offload)",
            }
        else if (tested_profile_match)
            "too-large"
        else
            "hidden";
        const fit_label = switch (fit.fit_state) {
            .fits => "yes",
            .fits_with_offload => "offload",
            .does_not_fit => "no",
        };
        var notes_buf: [80]u8 = undefined;
        const notes = if (low_vram_ctx != null and supported_now)
            try std.fmt.bufPrint(&notes_buf, "ROCm · {d}-token context · one request slot", .{low_vram_ctx.?})
        else if (supported_with_offload)
            "auto: MoE experts → host RAM (slower)"
        else if (fit.exact)
            "tested + exact fit"
        else
            "tested + catalog fit";
        try stdout.interface.print(
            "{s: <30} {s: <12} {s: <19} {s: <9} {s: <11} {s: <8} {s}\n",
            .{
                entry.id,
                entry.release_date,
                status_label,
                fit_label,
                if (installed) "yes" else "no",
                if (is_active) "yes" else "no",
                notes,
            },
        );
    }

    if (!rendered_any) {
        try stdout.interface.writeAll("No managed models are currently marked supported and fitting for this GPU profile.\n");
    }

    if (any_requires_offload) {
        try stdout.interface.writeAll("\nModels marked \"supported (offload)\" exceed your VRAM budget; the loader\n");
        try stdout.interface.writeAll("automatically routes MoE expert tensors to host RAM so they fit. Decode\n");
        try stdout.interface.writeAll("speed drops because experts are read over PCIe. Set ZINC_OFFLOAD_MOE_EXPERTS=0\n");
        try stdout.interface.writeAll("to opt out (the load will then OOM if the model doesn't fit otherwise).\n");
    }

    try stdout.interface.flush();
}

/// Format a byte count as a human-readable size string (e.g. "4.58 GiB").
fn formatSizeHuman(buf: *[32]u8, size_bytes: u64) []const u8 {
    const gib: f64 = @as(f64, @floatFromInt(size_bytes)) / (1024.0 * 1024.0 * 1024.0);
    if (gib >= 1.0) {
        return std.fmt.bufPrint(buf, "{d:.2} GiB", .{gib}) catch "??";
    }
    const mib: f64 = @as(f64, @floatFromInt(size_bytes)) / (1024.0 * 1024.0);
    return std.fmt.bufPrint(buf, "{d:.2} MiB", .{mib}) catch "??";
}

/// Write a JSON string value, escaping special characters.
fn writeJsonString(w: anytype, s: []const u8) !void {
    try w.writeAll("\"");
    for (s) |c| {
        switch (c) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            else => {
                const byte = [1]u8{c};
                try w.writeAll(&byte);
            },
        }
    }
    try w.writeAll("\"");
}

/// Output the full catalog as pretty-printed JSON.
/// When `support` is null, GPU detection was unavailable; `fits_gpu` is null and `requires_offload_to_fit` is false.
fn printManagedModelListJson(
    w: anytype,
    active_model_id: ?[]const u8,
    support: ?*const ManagedGpuSupport,
    allocator: std.mem.Allocator,
) !void {
    // Sort entries by id alphabetically.
    var sorted_indices: [catalog_mod.entries.len]usize = undefined;
    for (&sorted_indices, 0..) |*slot, idx| slot.* = idx;
    std.mem.sort(usize, &sorted_indices, {}, struct {
        fn lessThan(_: void, a: usize, b: usize) bool {
            return std.mem.order(u8, catalog_mod.entries[a].id, catalog_mod.entries[b].id) == .lt;
        }
    }.lessThan);

    try w.writeAll("[\n");
    var first = true;
    for (sorted_indices) |idx| {
        const entry = catalog_mod.entries[idx];
        if (!first) try w.writeAll(",\n");
        first = false;

        const installed = managed_mod.isInstalled(entry.id, allocator);
        const is_active = active_model_id != null and std.mem.eql(u8, active_model_id.?, entry.id);

        var size_buf: [32]u8 = undefined;
        const size_human = formatSizeHuman(&size_buf, entry.size_bytes);

        const status_str = @tagName(entry.status);

        const fit = if (support) |s| s.describeFit(entry, allocator) else null;
        const fits_gpu = if (fit) |f| f.fits_current_gpu else null;
        const low_vram_ctx = if (support) |s| s.rocm16GbContext(entry) else null;

        try w.writeAll("  {\n");

        try w.writeAll("    \"id\": ");
        try writeJsonString(w, entry.id);
        try w.writeAll(",\n");

        try w.writeAll("    \"display_name\": ");
        try writeJsonString(w, entry.display_name);
        try w.writeAll(",\n");

        try w.writeAll("    \"family\": ");
        try writeJsonString(w, entry.family);
        try w.writeAll(",\n");

        try w.writeAll("    \"release_date\": ");
        try writeJsonString(w, entry.release_date);
        try w.writeAll(",\n");

        try w.writeAll("    \"format\": ");
        try writeJsonString(w, entry.format);
        try w.writeAll(",\n");

        try w.writeAll("    \"quantization\": ");
        try writeJsonString(w, entry.quantization);
        try w.writeAll(",\n");

        try w.writeAll("    \"file_name\": ");
        try writeJsonString(w, entry.file_name);
        try w.writeAll(",\n");

        var sz_str_buf: [24]u8 = undefined;
        const size_bytes_str = std.fmt.bufPrint(&sz_str_buf, "{d}", .{entry.size_bytes}) catch "0";
        try w.writeAll("    \"size_bytes\": ");
        try w.writeAll(size_bytes_str);
        try w.writeAll(",\n");

        try w.writeAll("    \"size_human\": ");
        try writeJsonString(w, size_human);
        try w.writeAll(",\n");

        var vram_str_buf: [24]u8 = undefined;
        const required_vram = if (low_vram_ctx != null) catalog_mod.rocm_16gb_min_vram_bytes else entry.required_vram_bytes;
        const vram_str = std.fmt.bufPrint(&vram_str_buf, "{d}", .{required_vram}) catch "0";
        try w.writeAll("    \"required_vram_bytes\": ");
        try w.writeAll(vram_str);
        try w.writeAll(",\n");

        var vram_off_str_buf: [24]u8 = undefined;
        const required_with_offload = if (low_vram_ctx != null) required_vram else catalog_mod.requiredVramWithOffload(entry);
        const vram_off_str = std.fmt.bufPrint(&vram_off_str_buf, "{d}", .{required_with_offload}) catch "0";
        try w.writeAll("    \"required_vram_with_offload_bytes\": ");
        try w.writeAll(vram_off_str);
        try w.writeAll(",\n");

        const requires_offload = if (fit) |f| f.fit_state == .fits_with_offload else false;
        try w.writeAll("    \"requires_offload_to_fit\": ");
        try w.writeAll(if (requires_offload) "true" else "false");
        try w.writeAll(",\n");

        var ctx_str_buf: [12]u8 = undefined;
        const ctx_str = std.fmt.bufPrint(&ctx_str_buf, "{d}", .{low_vram_ctx orelse entry.default_context_length}) catch "0";
        try w.writeAll("    \"default_context_length\": ");
        try w.writeAll(ctx_str);
        try w.writeAll(",\n");

        try w.writeAll("    \"homepage_url\": ");
        try writeJsonString(w, entry.homepage_url);
        try w.writeAll(",\n");

        try w.writeAll("    \"download_url\": ");
        try writeJsonString(w, entry.download_url);
        try w.writeAll(",\n");

        try w.writeAll("    \"sha256\": ");
        try writeJsonString(w, entry.sha256);
        try w.writeAll(",\n");

        try w.writeAll("    \"status\": ");
        try writeJsonString(w, status_str);
        try w.writeAll(",\n");

        try w.writeAll("    \"recommended_for_chat\": ");
        try w.writeAll(if (entry.recommended_for_chat) "true" else "false");
        try w.writeAll(",\n");

        try w.writeAll("    \"thinking_stable\": ");
        try w.writeAll(if (entry.thinking_stable) "true" else "false");
        try w.writeAll(",\n");

        try w.writeAll("    \"tested_profiles\": [");
        for (entry.tested_profiles, 0..) |profile, pi| {
            if (pi > 0) try w.writeAll(", ");
            try writeJsonString(w, profile);
        }
        try w.writeAll("],\n");

        try w.writeAll("    \"installed\": ");
        try w.writeAll(if (installed) "true" else "false");
        try w.writeAll(",\n");

        try w.writeAll("    \"active\": ");
        try w.writeAll(if (is_active) "true" else "false");
        try w.writeAll(",\n");

        try w.writeAll("    \"fits_gpu\": ");
        if (fits_gpu) |fits| {
            try w.writeAll(if (fits) "true" else "false");
        } else {
            try w.writeAll("null");
        }
        try w.writeAll("\n");

        try w.writeAll("  }");
    }
    try w.writeAll("\n]\n");
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

            var support = try resolveManagedGpuSupport(config.gpuDevicePreference(), allocator);
            defer support.deinit(allocator);

            if (!catalog_mod.supportsProfile(entry.*, support.profile)) {
                // A detected-but-untested GPU profile (e.g. a Strix Halo APU or a
                // brand-new board) is a soft signal, not a hard incompatibility:
                // `pull` only downloads weights, and activation is still worth
                // attempting. Warn instead of refusing; for `use`, require --force
                // so running an unvalidated path stays an explicit choice.
                var warn_buffer: [1024]u8 = undefined;
                var warn_writer = std.fs.File.stderr().writerStreaming(&warn_buffer);
                const warn = &warn_writer.interface;
                try warn.print(
                    "warning: {s} has not been validated on this GPU profile ({s}).\n",
                    .{ entry.display_name, support.profile },
                );
                if (config.command == .model_use and !config.command_force) {
                    try warn.writeAll("Re-run with --force to activate it anyway, or run the file directly with -m <model.gguf>.\n");
                    try warn.flush();
                    return error.ModelUnsupportedOnThisGpu;
                }
                try warn.writeAll("Proceeding anyway; performance and correctness are unverified on this GPU.\n");
                try warn.flush();
            }

            if (config.command == .model_pull) {
                var stdout_buffer: [4096]u8 = undefined;
                var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);
                try managed_mod.pullModel(entry.*, allocator, &stdout.interface);
                try stdout.interface.flush();
                return;
            }

            if (!managed_mod.isInstalled(model_id, allocator)) return error.ModelNotInstalled;
            const fit = try managed_mod.verifyActiveSelectionFits(model_id, support.vram_budget_bytes, allocator);
            if (fit.fit_state == .does_not_fit) return error.ModelDoesNotFit;
            try managed_mod.writeActiveSelection(model_id, allocator);

            var stdout_buffer: [1024]u8 = undefined;
            var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);
            try stdout.interface.print("Active model set to {s}\n", .{model_id});
            if (fit.fit_state == .fits_with_offload) {
                try stdout.interface.writeAll("Note: this model exceeds your VRAM budget. The loader will automatically\n");
                try stdout.interface.writeAll("route MoE expert tensors to host RAM so it fits. Expect slower decode.\n");
            }
            try stdout.interface.flush();
        },
        .model_rm => {
            const model_id = config.command_model_id orelse return error.MissingArgValue;
            // `-hf` downloads share the managed cache but have no catalog
            // entry, so accept any id that is installed on disk and only
            // reject ids that are neither in the catalog nor installed.
            if (catalog_mod.find(model_id) == null and !managed_mod.isInstalled(model_id, allocator)) {
                return error.UnknownManagedModel;
            }

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
        .run, .chat => {},
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

// Accelerator prompt/server entrypoint shared by the CUDA and ROCm backends.
// Only referenced (and therefore only analyzed) for those backends. Does a real greedy decode:
// load → forward init → tokenize → prefill (one decodeStep per prompt token) →
// greedy generate (feed argmax back) → detokenize → print, or start the
// accelerator HTTP server with model lifecycle management.
fn runCuda(config: Config, resolved_path: ?[]const u8, allocator: std.mem.Allocator) !void {
    const model_path = resolved_path orelse {
        log.err("{s} backend requires a startup model", .{accelerator_name});
        return error.NoModelSpecified;
    };

    var device = try cuda_device_mod.CudaDevice.initBest(allocator);
    defer device.deinit();
    var name_buf: [256]u8 = undefined;
    if (comptime gpu.is_rocm) {
        log.info("ZINC ROCm backend — {s} ({d} CUs)", .{ device.name(&name_buf), device.smCount() });
    } else {
        log.info("ZINC CUDA backend — {s} (sm_{d}, {d} SMs)", .{
            device.name(&name_buf), device.computeCapability(), device.smCount(),
        });
    }

    // The server owns its model resources so unload and activation can replace
    // them while retaining the same selected device and HTTP listener.
    const entry = catalog_mod.findForLoadedModel(config.model_id, model_path, "");
    const default_ctx: u32 = if (entry != null and std.mem.eql(u8, entry.?.family, "inkling2"))
        entry.?.default_context_length
    else
        2048;
    const max_ctx: u32 = config.context_length orelse default_ctx;
    if (config.prompt == null) return runCudaServe(&device, model_path, config, max_ctx, allocator);

    // Load the model onto the GPU (weights uploaded verbatim-quantized).
    var model = loader_cuda_mod.Model.load(allocator, device.ctx, model_path) catch |err| {
        log.err("Failed to load model: {s}", .{@errorName(err)});
        return err;
    };
    defer model.deinit();

    if (comptime gpu.is_rocm) {
        if (model.config.architecture == .inkling2_loop) {
            var fwd = try forward_inkling_mod.ForwardInkling.init(allocator, &model, max_ctx);
            defer fwd.deinit();
            return runCudaDecode(&fwd, &model, config, max_ctx, allocator);
        }
    }

    // server_mode = no CLI prompt → multi-tenant HTTP serving (Effort 28). Both the
    // gemma4 dense forward AND the qwen35/36 hybrid-SSM+MoE forward expose the
    // batched serving path (decodeBatch + slot state); the server dispatches by
    // architecture via cuda_serve.Forward. Prompt mode (single-sequence) unchanged.

    // Build the forward state. max_ctx must cover prompt + generated tokens.
    // gemma4 and Muse Glimmer use the dense transformer forward path
    // (forward_cuda_gemma.zig); the
    // qwen35/qwen36 hybrid-SSM family uses forward_cuda.zig.
    if (model.config.architecture == .gemma or model.config.architecture == .muse_glimmer) {
        var fwd = forward_cuda_gemma_mod.ForwardGemma.init(allocator, &model, max_ctx) catch |err| {
            log.err("Failed to init {s} forward pass: {s}", .{ accelerator_name, @errorName(err) });
            return err;
        };
        defer fwd.deinit();
        log.info("{s} {s} forward init OK (n_embd={d}, n_layers={d}, vocab={d}, max_ctx={d})", .{
            accelerator_name, @tagName(model.config.architecture), fwd.d.n_embd, fwd.d.n_layers, fwd.d.vocab, max_ctx,
        });
        return runCudaDecode(&fwd, &model, config, max_ctx, allocator);
    }

    var fwd = forward_cuda_mod.ForwardCuda.init(allocator, &model, max_ctx) catch |err| {
        log.err("Failed to init {s} forward pass: {s}", .{ accelerator_name, @errorName(err) });
        return err;
    };
    defer fwd.deinit();
    log.info("{s} forward init OK (n_embd={d}, n_layers={d}, vocab={d}, max_ctx={d})", .{
        accelerator_name, fwd.d.n_embd, fwd.d.n_layers, fwd.d.vocab, max_ctx,
    });
    return runCudaDecode(&fwd, &model, config, max_ctx, allocator);
}

/// Effort 25: batched-GEMM prefill is the DEFAULT for the gemma forwards
/// (ForwardGemma exposes `prefillBatched`; the call site gates it at comptime so
/// the qwen forward — which has none — stays per-token regardless). Returns true
/// unless ZINC_BATCHED_PREFILL is explicitly set to an off value (0/off/false/no),
/// the opt-OUT back to the per-token prefill loop for debugging. Unset → on.
fn batchedPrefillDefaultOn() bool {
    const v = std.posix.getenv("ZINC_BATCHED_PREFILL") orelse return true;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

/// Tokenize the prompt, prefill, greedily generate, and print — shared by the
/// qwen and gemma CUDA forward paths (`fwd` is duck-typed: needs `decodeStep`
/// and `d.vocab`).
fn runCudaDecode(fwd: anytype, model: *loader_cuda_mod.Model, config: Config, max_ctx: u32, allocator: std.mem.Allocator) !void {
    const prompt = config.prompt.?;

    // Tokenizer is backend-agnostic — built from the GGUF metadata.
    var tokenizer = tokenizer_mod.Tokenizer.initFromGGUF(&model.gguf_file, allocator) catch |err| {
        log.err("Failed to init tokenizer from GGUF: {s}", .{@errorName(err)});
        return err;
    };
    defer tokenizer.deinit();

    const auto_chat = !config.chat and !config.raw_prompt and shouldAutoChatCliPrompt(&tokenizer, prompt);
    const use_chat_prompt = config.chat or auto_chat;
    var prepared_prompt = try prepareCliPrompt(&tokenizer, prompt, use_chat_prompt, config.system_prompt, allocator);
    defer prepared_prompt.deinit(allocator);

    const prompt_tokens = try tokenizer.encodePrompt(prepared_prompt.text, allocator);
    defer allocator.free(prompt_tokens);
    if (prompt_tokens.len == 0) {
        log.err("Prompt tokenized to zero tokens.", .{});
        return error.EmptyPrompt;
    }

    log.info("Prompt: {s}", .{prompt});
    log.info("Prompt tokens ({d}): {any}", .{ prompt_tokens.len, prompt_tokens[0..@min(prompt_tokens.len, 30)] });

    const eos_id = cliGenerationStopId(&tokenizer, use_chat_prompt);
    const max_new: u32 = config.max_tokens;

    // Clamp generation so prompt + generated never exceeds the KV/SSM context.
    if (prompt_tokens.len >= max_ctx) {
        log.err("Prompt ({d} tokens) does not fit in context ({d}); pass -c <larger>.", .{ prompt_tokens.len, max_ctx });
        return error.ContextTooSmall;
    }

    var generated: std.ArrayList(u32) = .{};
    defer generated.deinit(allocator);

    // MTP rollback/history buffers are execution-context setup, so allocate
    // them before prompt timing just as a separate draft context would be.
    var mtp_prepared = false;
    if (comptime @hasDecl(@TypeOf(fwd.*), "mtpPrepare")) {
        mtp_prepared = fwd.mtpPrepare() catch |err| blk: {
            log.warn("NextN/MTP context setup failed ({s}); using ordinary greedy decode", .{@errorName(err)});
            break :blk false;
        };
    }

    // PREFILL: build the KV cache for every prompt token, but only the LAST
    // prompt token needs logits — its argmax is the first generated token.
    // Prompt-internal tokens use prefillStep, which skips the vocab-sized LM
    // head (pure waste otherwise — a large share for the MoE models, whose
    // active per-token forward is small). Same KV cache, bit-identical output.
    var pos: u32 = 0;
    var next_tok: u32 = 0;
    var prefill_timer = try std.time.Timer.start();
    // Effort 25: batched-GEMM prefill is now the DEFAULT for gemma — it runs the
    // whole prompt through the batched path (gemma dense + MoE attention/FFN; the
    // method handles MoE internally) and is output-identical to the per-token
    // loop but ~4.6–4.7× faster. ZINC_BATCHED_PREFILL=0/off opts back out to the
    // per-token loop. Gated at comptime so the qwen forward (no prefillBatched)
    // stays per-token and still compiles.
    var used_batched = false;
    if (comptime @hasDecl(@TypeOf(fwd.*), "prefillBatched")) {
        if (batchedPrefillDefaultOn() and prompt_tokens.len > 1) {
            // Default-on path: degrade gracefully to the per-token loop if the
            // batched path fails (e.g. a large-prompt scratch allocation on a
            // memory-tight box) instead of aborting the whole run. The happy
            // path is byte-for-byte unchanged; a fallback is LOGGED so a real
            // regression on short prompts stays visible in the validate gate.
            // Mirrors the dbg_cuda harness, which already falls back on error.
            if (fwd.prefillBatched(prompt_tokens)) |first| {
                next_tok = first;
                pos = @intCast(prompt_tokens.len);
                used_batched = true;
            } else |err| {
                log.warn("batched prefill failed ({s}); falling back to per-token", .{@errorName(err)});
            }
        }
    }
    if (!used_batched) {
        while (pos < prompt_tokens.len) : (pos += 1) {
            if (pos + 1 < prompt_tokens.len) {
                try fwd.prefillStep(prompt_tokens[pos], pos);
            } else {
                next_tok = try fwd.decodeStep(prompt_tokens[pos], pos, true);
            }
        }
    }
    var mtp_active = false;
    if (comptime @hasDecl(@TypeOf(fwd.*), "mtpPrime")) {
        if (mtp_prepared and used_batched) {
            mtp_active = fwd.mtpPrime(prompt_tokens) catch |err| blk: {
                log.warn("NextN/MTP initialization failed ({s}); using ordinary greedy decode", .{@errorName(err)});
                break :blk false;
            };
        }
    }
    const prefill_ms = @as(f64, @floatFromInt(prefill_timer.read())) / 1_000_000.0;
    const prefill_tps = if (prefill_ms > 0) @as(f64, @floatFromInt(prompt_tokens.len)) / (prefill_ms / 1000.0) else 0;
    log.info("Prefill complete: {d} tokens in {d:.1} ms ({d:.2} tok/s)", .{ prompt_tokens.len, prefill_ms, prefill_tps });

    // GENERATE: greedily feed the argmax back until EOS or the token budget /
    // context limit is reached.
    var produced: u32 = 0;
    var decode_timer = try std.time.Timer.start();
    var generated_with_mtp = false;
    var mtp_drafted: u32 = 0;
    var mtp_accepted: u32 = 0;
    if (comptime @hasDecl(@TypeOf(fwd.*), "mtpCycle")) {
        if (mtp_active) {
            generated_with_mtp = true;
            mtp_loop: while (produced < max_new and pos < max_ctx) {
                if (next_tok == eos_id) break;

                const token_room = max_new - produced;
                const context_room = max_ctx - pos;
                const max_drafts = @min(@as(u32, 3), @min(token_room - 1, context_room - 1));
                if (max_drafts == 0) {
                    try generated.append(allocator, next_tok);
                    produced += 1;
                    break;
                }

                const result = try fwd.mtpCycle(next_tok, pos, max_drafts, eos_id);
                mtp_drafted += result.n_drafted;
                mtp_accepted += result.n_accepted;

                // The target always consumed and committed the seed token.
                try generated.append(allocator, next_tok);
                produced += 1;
                pos += 1;

                var i: u32 = 0;
                while (i < result.n_accepted) : (i += 1) {
                    const accepted = result.drafts[i];
                    if (accepted == eos_id) break :mtp_loop;
                    try generated.append(allocator, accepted);
                    produced += 1;
                    pos += 1;
                }
                next_tok = result.next_token;
            }
        }
    }
    if (!generated_with_mtp) {
        while (produced < max_new and pos < max_ctx) {
            if (next_tok == eos_id) break;
            try generated.append(allocator, next_tok);
            produced += 1;
            if (produced >= max_new or pos >= max_ctx) break;
            const fed = next_tok;
            next_tok = try fwd.decodeStep(fed, pos, true);
            pos += 1;
        }
    } else {
        const acceptance = if (mtp_drafted > 0)
            100.0 * @as(f64, @floatFromInt(mtp_accepted)) / @as(f64, @floatFromInt(mtp_drafted))
        else
            0.0;
        log.info("NextN/MTP: accepted {d}/{d} draft tokens ({d:.1}%)", .{ mtp_accepted, mtp_drafted, acceptance });
        if (comptime @hasDecl(@TypeOf(fwd.*), "mtpPerf")) {
            const perf = fwd.mtpPerf();
            log.info("NextN/MTP phases ({d} cycles): draft={d:.1} ms target={d:.1} ms restore={d:.1} ms catch-up={d:.1} ms", .{
                perf.cycles, perf.draft_ms, perf.target_ms, perf.restore_ms, perf.catchup_ms,
            });
        }
    }
    const decode_ms = @as(f64, @floatFromInt(decode_timer.read())) / 1_000_000.0;
    const decode_tps = if (decode_ms > 0) @as(f64, @floatFromInt(produced)) / (decode_ms / 1000.0) else 0;
    const ms_per_token = if (produced > 0) decode_ms / @as(f64, @floatFromInt(produced)) else 0;
    log.info("Generated {d} tokens in {d:.1} ms — {d:.2} tok/s ({d:.2} ms/tok)", .{ produced, decode_ms, decode_tps, ms_per_token });

    if (config.debug) {
        log.debug("Generated tokens ({d}): {any}", .{
            generated.items.len, generated.items[0..@min(generated.items.len, 20)],
        });
    }

    // Detokenize → text.
    var text_buf: std.ArrayList(u8) = .{};
    defer text_buf.deinit(allocator);
    for (generated.items) |tid| {
        var dec_buf: [256]u8 = undefined;
        const decoded = tokenizer.decodeToken(tid, &dec_buf);
        if (decoded.len > 0) {
            try text_buf.appendSlice(allocator, decoded);
        } else {
            try text_buf.appendSlice(allocator, "<?>");
        }
    }
    const template_kind = tokenizer.detectTemplateKind();
    if (use_chat_prompt and (template_kind == .inkling2 or template_kind == .atem)) {
        const normalized = try allocator.alloc(u8, text_buf.items.len);
        defer allocator.free(normalized);
        const text = try routes_mod.normalizeStructuredAssistantOutput(&tokenizer, text_buf.items, template_kind == .inkling2, normalized);
        log.info("Output ({d} tokens): {s}", .{ generated.items.len, text });
    } else {
        log.info("Output ({d} tokens): {s}", .{ generated.items.len, trimCliOutputText(text_buf.items, use_chat_prompt) });
    }
}

// ── Effort 28 increment 3 (3b): CUDA multi-tenant HTTP/SSE serving ───────────
//
// `zinc -m gemma.gguf` (no --prompt) → multi-tenant server: ONE GPU worker thread
// runs the continuous-batching loop (ServeEngine, src/server/cuda_serve.zig);
// each HTTP connection gets its own handler thread that tokenizes, enqueues, and
// SSE-streams its OWN tokens as the worker produces them. The serving path reuses
// the batched `decodeBatch` proven token-identical to N isolated single-sequence
// runs in increments 1+2; the production single-sequence prompt path is untouched.

/// A loaded accelerator model owns every object used by its worker. Heap
/// addresses stay stable while handler leases protect them from replacement.
const AcceleratorResources = struct {
    allocator: std.mem.Allocator,
    entry: *const accelerator_catalog.Entry,
    model: ?*loader_cuda_mod.Model = null,
    fwd: ?cuda_serve_mod.Forward = null,
    tokenizer: ?*tokenizer_mod.Tokenizer = null,
    engine: ?*cuda_serve_mod.ServeEngine = null,
    debug_ids: bool,
    slot_ctx: u32,

    fn load(allocator: std.mem.Allocator, device: *cuda_device_mod.CudaDevice, entry: *const accelerator_catalog.Entry, config: Config, max_ctx: u32) !*AcceleratorResources {
        const self = try allocator.create(AcceleratorResources);
        const slot_ctx = if (config.context_length == null)
            @min(max_ctx, entry.rocm_16gb_context_length orelse max_ctx)
        else
            max_ctx;
        self.* = .{ .allocator = allocator, .entry = entry, .debug_ids = envIsOn("ZINC_SERVE_DEBUG_IDS"), .slot_ctx = slot_ctx };
        errdefer self.destroy(device);
        const model = try allocator.create(loader_cuda_mod.Model);
        model.* = loader_cuda_mod.Model.load(allocator, device.ctx, entry.path) catch |err| {
            allocator.destroy(model);
            return err;
        };
        self.model = model;
        if (gpu.is_rocm and model.config.architecture == .inkling2_loop) {
            const fwd = try allocator.create(forward_inkling_mod.ForwardInkling);
            fwd.* = forward_inkling_mod.ForwardInkling.init(allocator, model, slot_ctx) catch |err| {
                allocator.destroy(fwd);
                return err;
            };
            self.fwd = .{ .inkling = fwd };
        } else if (model.config.architecture == .gemma or model.config.architecture == .muse_glimmer) {
            const fwd = try allocator.create(forward_cuda_gemma_mod.ForwardGemma);
            fwd.* = forward_cuda_gemma_mod.ForwardGemma.init(allocator, model, slot_ctx) catch |err| {
                allocator.destroy(fwd);
                return err;
            };
            self.fwd = .{ .gemma = fwd };
        } else {
            const fwd = try allocator.create(forward_cuda_mod.ForwardCuda);
            fwd.* = forward_cuda_mod.ForwardCuda.init(allocator, model, slot_ctx) catch |err| {
                allocator.destroy(fwd);
                return err;
            };
            self.fwd = .{ .qwen = fwd };
        }
        const tokenizer = try allocator.create(tokenizer_mod.Tokenizer);
        tokenizer.* = tokenizer_mod.Tokenizer.initFromGGUF(&model.gguf_file, allocator) catch |err| {
            allocator.destroy(tokenizer);
            return err;
        };
        self.tokenizer = tokenizer;
        var eos = tokenizer.eosId();
        if (std.posix.getenv("ZINC_SCHED_EOS")) |v| eos = std.fmt.parseInt(u32, std.mem.trim(u8, v, " \n\r\t"), 10) catch eos;
        const eot = if (!self.debug_ids and tokenizer.detectTemplateKind() == .atem) tokenizer.eot_id else null;
        const gemma_moe = model.config.architecture == .gemma and model.config.n_experts > 0;
        const single_sequence = gemma_moe or model.config.architecture == .inkling2_loop or model.constrained_vram;
        const nslots = cudaServeSlotCount(config.max_parallel, single_sequence);
        if (single_sequence and config.max_parallel != 1)
            log.warn("{s} serving uses one request slot, ignoring --parallel {d}", .{ @tagName(model.config.architecture), config.max_parallel });
        const engine = try allocator.create(cuda_serve_mod.ServeEngine);
        engine.* = cuda_serve_mod.ServeEngine.init(allocator, self.fwd.?, nslots, slot_ctx, eos, eot) catch |err| {
            allocator.destroy(engine);
            return err;
        };
        self.engine = engine;
        try engine.start();
        return self;
    }

    fn destroy(self: *AcceleratorResources, device: *cuda_device_mod.CudaDevice) void {
        // CUDA contexts are thread-local. The existing memory query selects this
        // retained context before releasing buffers from a handler thread.
        _ = device.freeMemory();
        if (self.engine) |engine| {
            engine.shutdown();
            engine.deinit();
            self.allocator.destroy(engine);
        }
        if (self.tokenizer) |tokenizer| {
            tokenizer.deinit();
            self.allocator.destroy(tokenizer);
        }
        if (self.fwd) |fwd| switch (fwd) {
            .gemma => |ptr| {
                ptr.deinit();
                self.allocator.destroy(ptr);
            },
            .qwen => |ptr| {
                ptr.deinit();
                self.allocator.destroy(ptr);
            },
            .inkling => |ptr| if (comptime gpu.is_rocm) {
                ptr.deinit();
                self.allocator.destroy(ptr);
            } else unreachable,
        };
        if (self.model) |model| {
            model.deinit();
            self.allocator.destroy(model);
        }
        self.allocator.destroy(self);
    }
};

const AcceleratorManager = struct {
    allocator: std.mem.Allocator,
    device: *cuda_device_mod.CudaDevice,
    config: Config,
    max_ctx: u32,
    inventory: accelerator_catalog.Inventory,
    current: ?*AcceleratorResources = null,
    /// Every generation holds a shared lease until its channel has drained.
    /// Exclusive lifecycle operations therefore never release in-use weights.
    lifecycle: std.Thread.RwLock = .{},

    fn deinit(self: *AcceleratorManager) void {
        if (self.current) |current| current.destroy(self.device);
        self.inventory.deinit();
    }

    fn capability(self: *const AcceleratorManager) cuda_serve_mod.MtpCapability {
        return if (self.current) |current| current.engine.?.mtpCapability() else .{ .unavailable_reason = "No model is loaded." };
    }

    pub fn unload(self: *AcceleratorManager) void {
        if (self.current) |current| current.destroy(self.device);
        self.current = null;
    }

    pub fn load(self: *AcceleratorManager, entry: *const accelerator_catalog.Entry) !*AcceleratorResources {
        return AcceleratorResources.load(self.allocator, self.device, entry, self.config, self.max_ctx);
    }

    fn activate(self: *AcceleratorManager, entry: *const accelerator_catalog.Entry) !void {
        return model_replacement.activate(self, entry);
    }
};

/// Per-connection context contains only stable manager state. A handler obtains
/// its current model pointers after acquiring a lifecycle lease.
const ServeConnCtx = struct {
    manager: *AcceleratorManager,
    conn: http_mod.Connection,
    allocator: std.mem.Allocator,
};

const ServeGenerateCtx = struct {
    engine: *cuda_serve_mod.ServeEngine,
    tokenizer: *tokenizer_mod.Tokenizer,
    allocator: std.mem.Allocator,
    debug_ids: bool,
    default_max: u32,
    slot_ctx: u32,
    model_path: ?[]const u8,
};

fn cudaServeSlotCount(requested: u32, single_sequence: bool) u32 {
    const clamped = std.math.clamp(requested, 1, 64);
    return if (single_sequence) 1 else clamped;
}

fn runCudaServe(device: *cuda_device_mod.CudaDevice, initial_path: []const u8, config: Config, max_ctx: u32, allocator: std.mem.Allocator) !void {
    var arch: [64]u8 = undefined;
    const capacity = device.totalMemory();
    const profile = if (gpu.is_rocm) catalog_mod.profileForRocm(device.arch(&arch), capacity) else null;
    var manager = AcceleratorManager{
        .allocator = allocator,
        .device = device,
        .config = config,
        .max_ctx = max_ctx,
        .inventory = try accelerator_catalog.Inventory.init(allocator, initial_path, capacity, profile),
    };
    defer manager.deinit();
    const initial_entry = manager.inventory.find(manager.inventory.initial_model_id) orelse return error.InitialModelMissing;
    try manager.activate(initial_entry);

    var server = http_mod.Server.init(allocator, config.port) catch |err| {
        log.err("Failed to bind HTTP server on port {d}: {s}", .{ config.port, @errorName(err) });
        return err;
    };
    defer server.deinit();
    log.info("ZINC {s} server listening on :{d} (ctx={d}, model lifecycle enabled)", .{ accelerator_name, config.port, manager.current.?.slot_ctx });
    while (true) {
        const conn = server.accept() catch |err| {
            log.warn("accept failed: {s}", .{@errorName(err)});
            continue;
        };
        const ctx = allocator.create(ServeConnCtx) catch {
            var c = conn;
            c.close();
            continue;
        };
        ctx.* = .{ .manager = &manager, .conn = conn, .allocator = allocator };
        const t = std.Thread.spawn(.{}, handleServeConn, .{ctx}) catch {
            ctx.conn.close();
            allocator.destroy(ctx);
            continue;
        };
        t.detach();
    }
}

/// Minimal request body the serve path reads (OpenAI-ish). Unknown fields ignored.
const ServeReqBody = struct {
    prompt: ?[]const u8 = null,
    messages: ?[]struct { role: []const u8 = "user", content: []const u8 = "" } = null,
    max_tokens: ?u32 = null,
    n_predict: ?u32 = null,
    /// Preserve the original SSE default when omitted; explicit false selects
    /// the OpenAI-compatible non-streaming response used by benchmark clients.
    stream: ?bool = null,
    /// Expose model reasoning in a `<think>` envelope when the template has a
    /// structured internal channel (including Muse Glimmer's ATEM protocol).
    enable_thinking: ?bool = null,
    enable_mtp: ?bool = null,
    model: ?[]const u8 = null,
};

/// Capability validation is separate from request policy: false and null are
/// always valid, while an explicit true needs this model/runtime to be ready.
fn serveMtpRequestError(enabled: ?bool, capability: anytype) ?[]const u8 {
    if ((enabled orelse false) and !capability.supports_mtp) return capability.unavailable_reason;
    return null;
}

fn buildCudaServeHealthJson(allocator: std.mem.Allocator, model_path: ?[]const u8, context_length: u32, capability: anytype) ![]u8 {
    const backend: []const u8 = if (gpu.is_rocm) "rocm" else "cuda";
    const name = if (model_path) |path| std.fs.path.basename(path) else "zinc";
    return std.json.Stringify.valueAlloc(allocator, .{
        .status = "ok",
        .backend = backend,
        .model = if (model_path == null) "" else name,
        .context_length = context_length,
        .supports_model_management = true,
        .supports_model_unload = true,
        .supports_model_pull = false,
        .supports_mtp = capability.supports_mtp,
        .mtp_default_enabled = capability.default_enabled,
        .mtp_request_control = true,
        .mtp_unavailable_reason = capability.unavailable_reason,
    }, .{});
}

fn buildCudaServeModelsJson(allocator: std.mem.Allocator, model_path: ?[]const u8, context_length: u32, supports_thinking: bool, capability: anytype) ![]u8 {
    const path = model_path orelse "zinc";
    const name = std.fs.path.basename(path);
    const catalog_entry = catalog_mod.findForLoadedModel(null, path, "");
    const entry = accelerator_catalog.Entry{
        .id = if (catalog_entry) |item| item.id else name,
        .display_name = if (catalog_entry) |item| item.display_name else name,
        .homepage_url = if (catalog_entry) |item| item.homepage_url else "",
    };
    return buildAcceleratorModelsJson(allocator, &.{entry}, entry.id, context_length, supports_thinking, capability);
}

/// Build discovery from the installed inventory, including unloaded candidates.
/// Paths remain internal; only an advertised opaque or catalog ID can activate.
fn buildAcceleratorModelsJson(allocator: std.mem.Allocator, entries: []const accelerator_catalog.Entry, active_id: ?[]const u8, context_length: u32, supports_thinking: bool, capability: anytype) ![]u8 {
    const Info = struct {
        id: []const u8,
        object: []const u8 = "model",
        display_name: []const u8,
        homepage_url: []const u8,
        family: []const u8,
        quantization: []const u8,
        size_bytes: u64,
        fit_source: []const u8,
        required_vram_bytes: u64,
        context_length: u32,
        installed: bool,
        active: bool,
        managed: bool,
        can_activate: bool,
        supported_on_current_gpu: bool,
        fits_current_gpu: bool,
        supports_thinking_toggle: bool,
        supports_mtp: bool,
        mtp_default_enabled: bool,
        mtp_request_control: bool = true,
        mtp_unavailable_reason: []const u8,
    };
    const data = try allocator.alloc(Info, entries.len);
    defer allocator.free(data);
    for (entries, data) |entry, *info| {
        const active = if (active_id) |id| std.mem.eql(u8, id, entry.id) else false;
        info.* = .{
            .id = entry.id,
            .display_name = entry.display_name,
            .homepage_url = entry.homepage_url,
            .family = entry.family,
            .quantization = entry.quantization,
            .size_bytes = entry.size_bytes,
            .required_vram_bytes = entry.required_vram_bytes,
            .fit_source = entry.fit_source,
            .context_length = if (active) context_length else 0,
            .installed = entry.installed,
            .active = active,
            .managed = entry.managed,
            .can_activate = entry.can_activate,
            .supported_on_current_gpu = entry.supported_on_current_gpu,
            .fits_current_gpu = active or entry.fits_current_gpu,
            .supports_thinking_toggle = active and supports_thinking,
            .supports_mtp = active and capability.supports_mtp,
            .mtp_default_enabled = active and capability.default_enabled,
            .mtp_unavailable_reason = if (active) capability.unavailable_reason else "Load this model to check MTP support.",
        };
    }
    return std.json.Stringify.valueAlloc(allocator, .{
        .object = "list",
        .supports_model_management = true,
        .supports_model_unload = true,
        .supports_model_pull = false,
        .data = data,
    }, .{});
}

const ServeUsage = struct {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
};

const ServeTimings = struct {
    prompt_n: usize,
    prompt_ms: f64,
    prompt_per_second: f64,
    predicted_n: usize,
    predicted_ms: f64,
    predicted_per_second: f64,
};

fn cudaServeTimings(prompt_tokens: usize, generated_tokens: usize, prefill_ns: u64, decode_ns: u64) ServeTimings {
    return .{
        .prompt_n = prompt_tokens,
        .prompt_ms = @as(f64, @floatFromInt(prefill_ns)) / 1_000_000.0,
        .prompt_per_second = if (prefill_ns == 0) 0 else @as(f64, @floatFromInt(prompt_tokens)) * 1_000_000_000.0 / @as(f64, @floatFromInt(prefill_ns)),
        .predicted_n = generated_tokens,
        .predicted_ms = @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0,
        .predicted_per_second = if (decode_ns == 0) 0 else @as(f64, @floatFromInt(generated_tokens)) * 1_000_000_000.0 / @as(f64, @floatFromInt(decode_ns)),
    };
}

fn cudaServeFinishReason(generated_tokens: usize, max_tokens: u32, last_token: ?u32, eos: u32, eot: ?u32) []const u8 {
    if (last_token) |token| {
        if (token == eos or (eot != null and token == eot.?)) return "stop";
    }
    return if (generated_tokens >= @as(usize, max_tokens)) "length" else "stop";
}

fn buildCudaServeFinishJson(allocator: std.mem.Allocator, is_chat: bool, finish_reason: []const u8, mtp: anytype, usage: ServeUsage, timings: ServeTimings) ![]u8 {
    if (is_chat) {
        const Choice = struct { index: u32 = 0, delta: struct {} = .{}, finish_reason: []const u8 };
        return std.json.Stringify.valueAlloc(allocator, .{
            .object = "chat.completion.chunk",
            .choices = [_]Choice{.{ .finish_reason = finish_reason }},
            .mtp = mtp,
            .usage = usage,
            .timings = timings,
        }, .{});
    }
    const Choice = struct { index: u32 = 0, text: []const u8 = "", finish_reason: []const u8 };
    return std.json.Stringify.valueAlloc(allocator, .{
        .object = "text_completion",
        .choices = [_]Choice{.{ .finish_reason = finish_reason }},
        .mtp = mtp,
        .usage = usage,
        .timings = timings,
    }, .{});
}

fn buildCudaServeResponseJson(allocator: std.mem.Allocator, is_chat: bool, content: []const u8, finish_reason: []const u8, mtp: anytype, usage: ServeUsage, timings: ServeTimings) ![]u8 {
    if (is_chat) {
        const Message = struct { role: []const u8 = "assistant", content: []const u8 };
        const Choice = struct { index: u32 = 0, message: Message, finish_reason: []const u8 };
        return std.json.Stringify.valueAlloc(allocator, .{
            .id = "chatcmpl-zinc",
            .object = "chat.completion",
            .choices = [_]Choice{.{ .message = .{ .content = content }, .finish_reason = finish_reason }},
            .usage = usage,
            .mtp = mtp,
            .timings = timings,
        }, .{});
    }
    const Choice = struct { index: u32 = 0, text: []const u8, finish_reason: []const u8 };
    return std.json.Stringify.valueAlloc(allocator, .{
        .id = "cmpl-zinc",
        .object = "text_completion",
        .choices = [_]Choice{.{ .text = content, .finish_reason = finish_reason }},
        .usage = usage,
        .mtp = mtp,
        .timings = timings,
    }, .{});
}

fn buildCudaServeErrorJson(allocator: std.mem.Allocator, message: []const u8, mtp: anytype) ![]u8 {
    return std.json.Stringify.valueAlloc(allocator, .{
        .@"error" = .{ .type = "server_error", .message = message },
        .mtp = mtp,
    }, .{});
}

const TestServeMtpCapability = struct {
    supports_mtp: bool,
    default_enabled: bool,
    unavailable_reason: []const u8,
};

test "MTP serving request validates explicit opt-in without disabling defaults" {
    const unsupported = TestServeMtpCapability{ .supports_mtp = false, .default_enabled = false, .unavailable_reason = "This model has no NextN weights" };
    try std.testing.expect(serveMtpRequestError(null, unsupported) == null);
    try std.testing.expect(serveMtpRequestError(false, unsupported) == null);
    const parsed = try std.json.parseFromSlice(ServeReqBody, std.testing.allocator, "{\"prompt\":\"hello\",\"enable_mtp\":true}", .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(unsupported.unavailable_reason, serveMtpRequestError(parsed.value.enable_mtp, unsupported).?);
    const supported = TestServeMtpCapability{ .supports_mtp = true, .default_enabled = true, .unavailable_reason = "" };
    try std.testing.expect(serveMtpRequestError(true, supported) == null);
    for ([_][]const u8{ "{\"prompt\":\"hello\"}", "{\"prompt\":\"hello\",\"enable_mtp\":null}" }) |body| {
        const request = try std.json.parseFromSlice(ServeReqBody, std.testing.allocator, body, .{});
        defer request.deinit();
        try std.testing.expect(request.value.enable_mtp == null);
    }
}

test "MTP serving health and model discovery report the same runtime capability" {
    const allocator = std.testing.allocator;
    for ([_]TestServeMtpCapability{
        .{ .supports_mtp = true, .default_enabled = true, .unavailable_reason = "" },
        .{ .supports_mtp = true, .default_enabled = false, .unavailable_reason = "" },
        .{ .supports_mtp = false, .default_enabled = false, .unavailable_reason = "MTP requires one request slot" },
    }) |capability| {
        const health = try buildCudaServeHealthJson(allocator, "/fixtures/Qwen3.8-27B-Q4_K_M.gguf", 8192, capability);
        defer allocator.free(health);
        const models = try buildCudaServeModelsJson(allocator, "/fixtures/Qwen3.8-27B-Q4_K_M.gguf", 8192, true, capability);
        defer allocator.free(models);
        var parsed_health = try std.json.parseFromSlice(std.json.Value, allocator, health, .{});
        defer parsed_health.deinit();
        var parsed_models = try std.json.parseFromSlice(std.json.Value, allocator, models, .{});
        defer parsed_models.deinit();
        const model = parsed_models.value.object.get("data").?.array.items[0];
        inline for (.{ "supports_mtp", "mtp_default_enabled", "mtp_request_control" }) |field| {
            try std.testing.expectEqual(parsed_health.value.object.get(field).?.bool, model.object.get(field).?.bool);
        }
        try std.testing.expectEqual(capability.supports_mtp, model.object.get("supports_mtp").?.bool);
        try std.testing.expectEqual(capability.default_enabled, model.object.get("mtp_default_enabled").?.bool);
        try std.testing.expectEqualStrings(capability.unavailable_reason, model.object.get("mtp_unavailable_reason").?.string);
        try std.testing.expect(model.object.get("active").?.bool);
        try std.testing.expect(model.object.get("supports_thinking_toggle").?.bool);
    }
}

test "accelerator model discovery preserves reload candidates after unload and refreshes MTP after switching" {
    const allocator = std.testing.allocator;
    const entries = [_]accelerator_catalog.Entry{
        .{ .id = "first", .display_name = "First model" },
        .{ .id = "second", .display_name = "Second model" },
    };
    const capability = TestServeMtpCapability{ .supports_mtp = true, .default_enabled = true, .unavailable_reason = "" };
    for ([_]?[]const u8{ "first", null, "second" }) |active_id| {
        const json = try buildAcceleratorModelsJson(allocator, &entries, active_id, 4096, true, capability);
        defer allocator.free(json);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
        defer parsed.deinit();
        const object = parsed.value.object;
        try std.testing.expect(object.get("supports_model_management").?.bool);
        try std.testing.expect(object.get("supports_model_unload").?.bool);
        try std.testing.expect(!object.get("supports_model_pull").?.bool);
        const data = object.get("data").?.array.items;
        try std.testing.expectEqual(@as(usize, 2), data.len);
        for (data) |item| {
            const model = item.object;
            const active = if (active_id) |id| std.mem.eql(u8, id, model.get("id").?.string) else false;
            try std.testing.expectEqual(active, model.get("active").?.bool);
            try std.testing.expectEqual(active, model.get("supports_mtp").?.bool);
            try std.testing.expectEqual(active, model.get("mtp_default_enabled").?.bool);
            try std.testing.expect(model.get("installed").?.bool);
            try std.testing.expect(model.get("can_activate").?.bool);
            try std.testing.expect(model.get("path") == null);
        }
    }
    const idle = try buildCudaServeHealthJson(allocator, null, 0, TestServeMtpCapability{ .supports_mtp = false, .default_enabled = false, .unavailable_reason = "No model is loaded." });
    defer allocator.free(idle);
    var parsed_idle = try std.json.parseFromSlice(std.json.Value, allocator, idle, .{});
    defer parsed_idle.deinit();
    try std.testing.expectEqualStrings("", parsed_idle.value.object.get("model").?.string);
    try std.testing.expect(!parsed_idle.value.object.get("supports_mtp").?.bool);
}

test "accelerator model requests reject stale selection after a switch" {
    try std.testing.expect(acceleratorModelRequestError(null, "active-id", "/fixtures/current.gguf") == null);
    try std.testing.expect(acceleratorModelRequestError("active-id", "active-id", "/fixtures/current.gguf") == null);
    try std.testing.expect(acceleratorModelRequestError("current.gguf", "active-id", "/fixtures/current.gguf") == null);
    try std.testing.expect(acceleratorModelRequestError("q", "active-id", "/fixtures/current.gguf") == null);
    try std.testing.expect(acceleratorModelRequestError("zinc", "active-id", "/fixtures/current.gguf") == null);
    try std.testing.expect(acceleratorModelRequestError("previous-id", "active-id", "/fixtures/current.gguf") != null);
    try std.testing.expect(acceleratorModelRequestError("../current.gguf", "active-id", "/fixtures/current.gguf") != null);
    try std.testing.expect(acceleratorModelRequestError("active-id", null, null) != null);
}

test "MTP terminal stream metadata preserves effective mode and actual fallback" {
    const allocator = std.testing.allocator;
    const result = .{ .requested = @as(?bool, null), .enabled = true, .used = false, .drafted = @as(u64, 0), .accepted = @as(u64, 0), .cycles = @as(u64, 0), .fallback_reason = @as(?[]const u8, "NextN priming failed") };
    for ([_]bool{ true, false }) |is_chat| {
        const usage = ServeUsage{ .prompt_tokens = 3, .completion_tokens = 2, .total_tokens = 5 };
        const json = try buildCudaServeFinishJson(allocator, is_chat, "stop", result, usage, cudaServeTimings(3, 2, 30_000_000, 20_000_000));
        defer allocator.free(json);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
        defer parsed.deinit();
        const mtp = parsed.value.object.get("mtp").?.object;
        try std.testing.expect(mtp.get("requested").? == .null);
        try std.testing.expect(mtp.get("enabled").?.bool);
        try std.testing.expect(!mtp.get("used").?.bool);
        try std.testing.expectEqualStrings("NextN priming failed", mtp.get("fallback_reason").?.string);
        const choice = parsed.value.object.get("choices").?.array.items[0];
        try std.testing.expectEqualStrings("stop", choice.object.get("finish_reason").?.string);
        try std.testing.expectEqual(@as(i64, 2), parsed.value.object.get("usage").?.object.get("completion_tokens").?.integer);
        const rate = parsed.value.object.get("timings").?.object.get("predicted_per_second").?;
        try std.testing.expectApproxEqAbs(@as(f64, 100), switch (rate) {
            .float => |value| value,
            .integer => |value| @as(f64, @floatFromInt(value)),
            else => return error.InvalidTimingNumber,
        }, 0.001);
    }
}

test "MTP serving response metadata describes emitted tokens and exact content" {
    const allocator = std.testing.allocator;
    const content = "Zig \"code\"\nline\t<&>";
    const result = .{ .requested = @as(?bool, false), .enabled = false, .used = false, .drafted = @as(u64, 0), .accepted = @as(u64, 0), .cycles = @as(u64, 0), .fallback_reason = @as(?[]const u8, null) };
    const usage = ServeUsage{ .prompt_tokens = 3, .completion_tokens = 2, .total_tokens = 5 };
    for ([_]bool{ true, false }) |is_chat| {
        const json = try buildCudaServeResponseJson(allocator, is_chat, content, "length", result, usage, cudaServeTimings(3, 2, 0, 0));
        defer allocator.free(json);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
        defer parsed.deinit();
        const choice = parsed.value.object.get("choices").?.array.items[0].object;
        const actual = if (is_chat) choice.get("message").?.object.get("content").?.string else choice.get("text").?.string;
        try std.testing.expectEqualStrings(content, actual);
        try std.testing.expect(!parsed.value.object.get("mtp").?.object.get("enabled").?.bool);
        try std.testing.expectEqual(@as(i64, 2), parsed.value.object.get("usage").?.object.get("completion_tokens").?.integer);
        const timings = parsed.value.object.get("timings").?.object;
        try std.testing.expectEqual(@as(i64, 2), timings.get("predicted_n").?.integer);
        const rate = timings.get("predicted_per_second").?;
        try std.testing.expectEqual(@as(f64, 0), switch (rate) {
            .float => |value| value,
            .integer => |value| @as(f64, @floatFromInt(value)),
            else => return error.InvalidTimingNumber,
        });
    }
}

test "MTP serving errors preserve execution status and explain fallback" {
    const allocator = std.testing.allocator;
    const result = .{ .requested = @as(?bool, true), .enabled = true, .used = false, .drafted = @as(u64, 0), .accepted = @as(u64, 0), .cycles = @as(u64, 0), .fallback_reason = @as(?[]const u8, "GPU execution failed") };
    const json = try buildCudaServeErrorJson(allocator, "GPU generation failed", result);
    defer allocator.free(json);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("GPU generation failed", parsed.value.object.get("error").?.object.get("message").?.string);
    try std.testing.expect(!parsed.value.object.get("mtp").?.object.get("used").?.bool);
    try std.testing.expectEqualStrings("GPU execution failed", parsed.value.object.get("mtp").?.object.get("fallback_reason").?.string);
}

test "MTP serving stop tokens take precedence over an exact output budget" {
    try std.testing.expectEqualStrings("stop", cudaServeFinishReason(1, 1, 2, 2, null));
    try std.testing.expectEqualStrings("stop", cudaServeFinishReason(4, 4, 10, 2, 10));
    try std.testing.expectEqualStrings("stop", cudaServeFinishReason(5, 5, 2, 2, 10));
    try std.testing.expectEqualStrings("length", cudaServeFinishReason(1, 1, 7, 2, 10));
    try std.testing.expectEqualStrings("length", cudaServeFinishReason(4, 4, 7, 2, null));
    try std.testing.expectEqualStrings("stop", cudaServeFinishReason(3, 4, 7, 2, null));
    try std.testing.expectEqualStrings("stop", cudaServeFinishReason(0, 1, null, 2, null));
}

const AcceleratorActionBody = struct {
    model: ?[]const u8 = null,
    model_id: ?[]const u8 = null,
};

fn tryBeginAcceleratorChange(lifecycle: *std.Thread.RwLock) error{ModelBusy}!void {
    if (!lifecycle.tryLock()) return error.ModelBusy;
}

test "accelerator lifecycle refuses model changes while a request owns a lease" {
    var lifecycle: std.Thread.RwLock = .{};
    lifecycle.lockShared();
    try std.testing.expectError(error.ModelBusy, tryBeginAcceleratorChange(&lifecycle));
    lifecycle.unlockShared();
    try tryBeginAcceleratorChange(&lifecycle);
    // No new generation may enter while model resources are being replaced.
    try std.testing.expect(!lifecycle.tryLockShared());
    lifecycle.unlock();
    lifecycle.lockShared();
    lifecycle.unlockShared();
}

fn acceleratorModelRequestError(requested: ?[]const u8, active_id: ?[]const u8, active_path: ?[]const u8) ?[]const u8 {
    const id = requested orelse return null;
    if (id.len == 0) return null;
    // Existing benchmark/OpenAI clients use these aliases for the active model.
    // Explicit advertised IDs still reject stale selections after a switch.
    if (active_id != null and (std.mem.eql(u8, id, "q") or std.mem.eql(u8, id, "zinc"))) return null;
    if (active_id) |current| if (std.mem.eql(u8, id, current)) return null;
    if (active_path) |path| if (std.mem.eql(u8, id, std.fs.path.basename(path))) return null;
    return "The requested model is not active. Refresh the model list and load it before generating.";
}

fn acceleratorUnloadTargetMatches(requested: ?[]const u8, active_id: ?[]const u8) bool {
    // Repeated unload is safe even if the original target is no longer loaded.
    const current = active_id orelse return true;
    const target = requested orelse return true;
    return target.len == 0 or std.mem.eql(u8, target, current);
}

fn buildAcceleratorUnloadJson(allocator: std.mem.Allocator, unloaded: bool) ![]u8 {
    return std.json.Stringify.valueAlloc(allocator, .{ .object = "model.unload", .unloaded_from_gpu = unloaded, .active = false, .deleted_model = false }, .{});
}

test "accelerator unload is idempotent and rejects only a different active target" {
    try std.testing.expect(acceleratorUnloadTargetMatches("previous", null));
    try std.testing.expect(acceleratorUnloadTargetMatches(null, null));
    try std.testing.expect(acceleratorUnloadTargetMatches(null, "current"));
    try std.testing.expect(acceleratorUnloadTargetMatches("", "current"));
    try std.testing.expect(acceleratorUnloadTargetMatches("current", "current"));
    try std.testing.expect(!acceleratorUnloadTargetMatches("previous", "current"));
    for ([_]bool{ true, false }) |unloaded| {
        const json = try buildAcceleratorUnloadJson(std.testing.allocator, unloaded);
        defer std.testing.allocator.free(json);
        var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
        defer parsed.deinit();
        const result = parsed.value.object;
        try std.testing.expectEqualStrings("model.unload", result.get("object").?.string);
        try std.testing.expectEqual(unloaded, result.get("unloaded_from_gpu").?.bool);
        try std.testing.expect(!result.get("active").?.bool);
        try std.testing.expect(!result.get("deleted_model").?.bool);
        try std.testing.expect(result.get("unloaded") == null);
    }
}

fn handleAcceleratorAction(manager: *AcceleratorManager, conn: *http_mod.Connection, body: []const u8, activate: bool, allocator: std.mem.Allocator) !void {
    const action_body = if (std.mem.trim(u8, body, " \r\n\t").len == 0) "{}" else body;
    const parsed = std.json.parseFromSlice(AcceleratorActionBody, allocator, action_body, .{ .ignore_unknown_fields = true }) catch {
        try conn.sendError(400, "invalid_request_error", "Invalid JSON in model action");
        return;
    };
    defer parsed.deinit();
    tryBeginAcceleratorChange(&manager.lifecycle) catch {
        try conn.sendError(409, "model_busy", "The model is in use. Wait for the current replies to finish.");
        return;
    };
    defer manager.lifecycle.unlock();
    const requested = parsed.value.model orelse parsed.value.model_id;
    if (activate) {
        const id = requested orelse {
            try conn.sendError(400, "invalid_request_error", "A model id is required");
            return;
        };
        const entry = manager.inventory.find(id) orelse {
            try conn.sendError(400, "invalid_request_error", "Unknown installed model id");
            return;
        };
        if (!entry.can_activate or !entry.supported_on_current_gpu or !entry.fits_current_gpu) {
            try conn.sendError(400, "invalid_request_error", "This model is not supported or does not fit the current GPU");
            return;
        }
        manager.activate(entry) catch |err| {
            log.warn("Model activation failed: {s}", .{@errorName(err)});
            try conn.sendError(500, "model_load_error", if (manager.current != null) "Could not load the model. The previous model was restored." else "Could not load the model. No model is loaded; select an installed model to try again.");
            return;
        };
        const response = try std.json.Stringify.valueAlloc(allocator, .{ .object = "model.activation", .id = entry.id, .active = true }, .{});
        defer allocator.free(response);
        try conn.sendJson(200, response);
    } else {
        if (!acceleratorUnloadTargetMatches(requested, if (manager.current) |current| current.entry.id else null)) {
            try conn.sendError(409, "model_changed", "The active model changed. Refresh before unloading.");
            return;
        }
        const unloaded = manager.current != null;
        manager.unload();
        const response = try buildAcceleratorUnloadJson(allocator, unloaded);
        defer allocator.free(response);
        try conn.sendJson(200, response);
    }
}

fn handleServeConn(ctx: *ServeConnCtx) void {
    defer ctx.allocator.destroy(ctx);
    var conn = ctx.conn;
    defer conn.close();
    const req = conn.readRequest() catch return;
    const manager = ctx.manager;
    if (req.method == .OPTIONS) {
        conn.sendJson(200, "{}") catch {};
        return;
    }
    if (req.method == .GET and (std.mem.eql(u8, req.path, "/") or std.mem.eql(u8, req.path, "/chat"))) {
        routes_mod.serveChatUi(&conn) catch {};
        return;
    }
    if (req.method == .POST and (std.mem.eql(u8, req.path, "/v1/models/activate") or std.mem.eql(u8, req.path, "/v1/models/unload"))) {
        handleAcceleratorAction(manager, &conn, req.body, std.mem.eql(u8, req.path, "/v1/models/activate"), ctx.allocator) catch |err| log.warn("Model action response failed: {s}", .{@errorName(err)});
        return;
    }
    if (req.method == .POST and std.mem.eql(u8, req.path, "/v1/chat/reset")) {
        const parsed = std.json.parseFromSlice(struct { session_id: []const u8 = "" }, ctx.allocator, req.body, .{ .ignore_unknown_fields = true }) catch {
            conn.sendError(400, "invalid_request_error", "Invalid chat reset request") catch {};
            return;
        };
        defer parsed.deinit();
        tryBeginAcceleratorChange(&manager.lifecycle) catch {
            conn.sendError(409, "model_busy", "Wait for the current replies to finish before resetting chat") catch {};
            return;
        };
        defer manager.lifecycle.unlock();
        // Accelerator serving resets KV/SSM at every request boundary and has no
        // cross-request prompt reuse. No live buffers need mutating here.
        conn.sendJson(200, "{\"object\":\"chat.reset\",\"reset\":true}") catch {};
        return;
    }
    manager.lifecycle.lockShared();
    defer manager.lifecycle.unlockShared();
    const current = manager.current;
    const capability = manager.capability();
    if (req.method == .GET and std.mem.eql(u8, req.path, "/health")) {
        const body = buildCudaServeHealthJson(ctx.allocator, if (current) |r| r.entry.path else null, if (current) |r| r.slot_ctx else 0, capability) catch {
            conn.sendError(500, "server_error", "Failed to describe runtime capability") catch {};
            return;
        };
        defer ctx.allocator.free(body);
        conn.sendJson(200, body) catch {};
        return;
    }
    if (req.method == .GET and std.mem.eql(u8, req.path, "/v1/models")) {
        const body = buildAcceleratorModelsJson(ctx.allocator, manager.inventory.entries.items, if (current) |r| r.entry.id else null, if (current) |r| r.slot_ctx else 0, if (current) |r| r.tokenizer.?.supportsThinkingToggle() else false, capability) catch {
            conn.sendError(500, "server_error", "Failed to describe installed models") catch {};
            return;
        };
        defer ctx.allocator.free(body);
        conn.sendJson(200, body) catch {};
        return;
    }
    if (req.method == .GET and std.mem.eql(u8, req.path, "/stats")) {
        var sbuf: [256]u8 = undefined;
        const js = if (current) |r| r.engine.?.statsJson(&sbuf) catch "{}" else "{}";
        conn.sendJson(200, js) catch {};
        return;
    }
    const is_gen = req.method == .POST and (std.mem.eql(u8, req.path, "/v1/completions") or std.mem.eql(u8, req.path, "/v1/chat/completions") or std.mem.eql(u8, req.path, "/generate"));
    if (!is_gen) {
        conn.sendError(404, "not_found", "unknown route") catch {};
        return;
    }
    const resource = current orelse {
        conn.sendError(503, "model_not_loaded", "No model is loaded. Select an installed model first.") catch {};
        return;
    };
    const parsed_model = std.json.parseFromSlice(struct { model: ?[]const u8 = null }, ctx.allocator, req.body, .{ .ignore_unknown_fields = true }) catch {
        conn.sendError(400, "invalid_request_error", "Invalid JSON request") catch {};
        return;
    };
    defer parsed_model.deinit();
    if (acceleratorModelRequestError(parsed_model.value.model, resource.entry.id, resource.entry.path)) |message| {
        conn.sendError(409, "model_changed", message) catch {};
        return;
    }
    var active = ServeGenerateCtx{
        .engine = resource.engine.?,
        .tokenizer = resource.tokenizer.?,
        .allocator = ctx.allocator,
        .debug_ids = resource.debug_ids,
        .default_max = manager.config.max_tokens,
        .slot_ctx = resource.slot_ctx,
        .model_path = resource.entry.path,
    };
    const is_chat = std.mem.eql(u8, req.path, "/v1/chat/completions");
    handleServeGenerate(&active, &conn, req.body, is_chat) catch |err| log.warn("serve request failed: {s}", .{@errorName(err)});
}

fn handleServeGenerate(ctx: *ServeGenerateCtx, conn: *http_mod.Connection, body: []const u8, is_chat: bool) !void {
    const allocator = ctx.allocator;

    // ── Build the prompt token ids ───────────────────────────────────────────
    var prompt_tokens: []u32 = undefined;
    var max_tokens: u32 = ctx.default_max;
    var owns_tokens = false;
    var stream_response = true;
    var enable_thinking: ?bool = null;
    var enable_mtp: ?bool = null;

    if (ctx.debug_ids) {
        // Gate contract: body is JSON {"prompt":"t0,t1,...","max_tokens":N}; the
        // prompt is raw comma-separated token ids — bypasses tokenizer + detok so
        // the streamed ids compare exactly to the isolated reference.
        const parsed = std.json.parseFromSlice(ServeReqBody, allocator, body, .{ .ignore_unknown_fields = true }) catch {
            try conn.sendError(400, "invalid_request_error", "bad json");
            return;
        };
        defer parsed.deinit();
        if (serveMtpRequestError(parsed.value.enable_mtp, ctx.engine.mtpCapability())) |message| {
            try conn.sendError(400, "invalid_request_error", message);
            return;
        }
        if (parsed.value.max_tokens orelse parsed.value.n_predict) |m| max_tokens = m;
        stream_response = parsed.value.stream orelse true;
        enable_thinking = parsed.value.enable_thinking;
        enable_mtp = parsed.value.enable_mtp;
        const ptxt = parsed.value.prompt orelse {
            try conn.sendError(400, "invalid_request_error", "missing prompt");
            return;
        };
        prompt_tokens = try parseTokenIdList(allocator, ptxt);
        owns_tokens = true;
    } else {
        const parsed = std.json.parseFromSlice(ServeReqBody, allocator, body, .{ .ignore_unknown_fields = true }) catch {
            try conn.sendError(400, "invalid_request_error", "bad json");
            return;
        };
        defer parsed.deinit();
        if (serveMtpRequestError(parsed.value.enable_mtp, ctx.engine.mtpCapability())) |message| {
            try conn.sendError(400, "invalid_request_error", message);
            return;
        }
        if (parsed.value.max_tokens orelse parsed.value.n_predict) |m| max_tokens = m;
        stream_response = parsed.value.stream orelse true;
        enable_thinking = parsed.value.enable_thinking;
        enable_mtp = parsed.value.enable_mtp;

        var prompt_text: []const u8 = "";
        var chat_buf: ?[]u8 = null;
        defer if (chat_buf) |b| allocator.free(b);
        if (is_chat) {
            const msgs = parsed.value.messages orelse {
                try conn.sendError(400, "invalid_request_error", "missing messages");
                return;
            };
            var roles = try allocator.alloc([]const u8, msgs.len);
            defer allocator.free(roles);
            var contents = try allocator.alloc([]const u8, msgs.len);
            defer allocator.free(contents);
            for (msgs, 0..) |m, i| {
                roles[i] = m.role;
                contents[i] = m.content;
            }
            const buf = try allocator.alloc(u8, 64 * 1024);
            chat_buf = buf;
            prompt_text = ctx.tokenizer.applyChatTemplateWithOptions(roles, contents, .{ .enable_thinking = enable_thinking }, buf) catch |err| {
                try conn.sendError(400, "invalid_request_error", "chat template failed");
                log.warn("chat template: {s}", .{@errorName(err)});
                return;
            };
        } else {
            prompt_text = parsed.value.prompt orelse {
                try conn.sendError(400, "invalid_request_error", "missing prompt");
                return;
            };
        }
        prompt_tokens = try ctx.tokenizer.encodePrompt(prompt_text, allocator);
        owns_tokens = true;
    }
    defer if (owns_tokens) allocator.free(prompt_tokens);

    const template_kind = ctx.tokenizer.detectTemplateKind();
    const structured_chat = is_chat and !ctx.debug_ids and
        (template_kind == .atem or template_kind == .openai_moe or template_kind == .inkling2);
    // ATEM models always begin on their private reasoning channel. Expose that
    // channel by default (callers can still request enable_thinking=false), or a
    // short completion that has not reached the final channel serializes as an
    // empty assistant message even though the model generated valid tokens.
    const thinking_enabled = structured_chat and (enable_thinking orelse (template_kind == .atem or template_kind == .inkling2)) and
        (ctx.tokenizer.supportsThinkingToggle() or template_kind == .atem);

    if (prompt_tokens.len == 0) {
        try conn.sendError(400, "invalid_request_error", "empty prompt");
        return;
    }
    // Clamp so prompt + generated never exceeds the slot KV depth.
    if (prompt_tokens.len >= ctx.slot_ctx) {
        try conn.sendError(400, "invalid_request_error", "prompt longer than context");
        return;
    }
    const room: u32 = ctx.slot_ctx - @as(u32, @intCast(prompt_tokens.len));
    if (max_tokens == 0 or max_tokens > room) max_tokens = room;

    const structured_capacity = @max(@as(usize, 64 * 1024), @as(usize, ctx.slot_ctx) * 32 + 256);
    const structured_buf: ?[]u8 = if (structured_chat) try allocator.alloc(u8, structured_capacity) else null;
    defer if (structured_buf) |buf_to_free| allocator.free(buf_to_free);

    // ── Submit + stream ──────────────────────────────────────────────────────
    var chan = cuda_serve_mod.ReqChannel{};
    const id = ctx.engine.submitWithOptions(prompt_tokens, max_tokens, &chan, .{ .enable_mtp = enable_mtp }) catch |err| {
        if (err == error.MtpUnavailable) {
            try conn.sendError(400, "invalid_request_error", ctx.engine.mtpCapability().unavailable_reason);
            return;
        }
        try conn.sendError(500, "server_error", "enqueue failed");
        return;
    };
    // From here the worker borrows `prompt_tokens` + `chan`; do not return without
    // draining to `finished` (so the worker is done touching them) then `finish`.
    defer {
        // Also covers any future allocation/write error added below: never
        // release a lifecycle lease while the worker still borrows this stack.
        drainQuietly(ctx.engine, &chan);
        ctx.engine.finish(id, &chan);
    }

    if (stream_response) {
        conn.sendSseStart() catch {
            // Client gone before headers — still drain the engine so it frees the slot.
            drainQuietly(ctx.engine, &chan);
            return;
        };
    }

    var buf: [32]u32 = undefined;
    var dec_buf: [256]u8 = undefined;
    var json_buf: [1024]u8 = undefined;
    var output: std.ArrayList(u8) = .{};
    defer output.deinit(allocator);
    var sent_structured_len: usize = 0;
    var write_failed = false;
    var generation_failed = false;
    var last_generated_token: ?u32 = null;
    while (true) {
        const ch = ctx.engine.nextChunk(&chan, &buf);
        if (ch.n > 0) last_generated_token = buf[ch.n - 1];
        generation_failed = generation_failed or ch.failed;
        var i: usize = 0;
        while (i < ch.n) : (i += 1) {
            const tok = buf[i];
            if (write_failed) continue; // keep draining the engine, stop writing
            if (is_chat and !ctx.debug_ids and (tok == ctx.engine.eos or tok == ctx.engine.eot)) continue;
            if (!stream_response) {
                if (ctx.debug_ids) {
                    if (output.items.len != 0) output.append(allocator, ',') catch {
                        write_failed = true;
                        continue;
                    };
                    output.writer(allocator).print("{d}", .{tok}) catch {
                        write_failed = true;
                        continue;
                    };
                } else {
                    output.appendSlice(allocator, ctx.tokenizer.decodeToken(tok, &dec_buf)) catch {
                        write_failed = true;
                        continue;
                    };
                }
            } else if (ctx.debug_ids) {
                const payload = std.fmt.bufPrint(&json_buf, "{d}", .{tok}) catch continue;
                conn.writeSseEvent(payload) catch {
                    write_failed = true;
                };
            } else if (structured_chat) {
                const text = ctx.tokenizer.decodeToken(tok, &dec_buf);
                output.appendSlice(allocator, text) catch {
                    write_failed = true;
                    continue;
                };
                const normalized = routes_mod.normalizeStructuredAssistantOutput(
                    ctx.tokenizer,
                    output.items,
                    thinking_enabled,
                    structured_buf.?,
                ) catch "";
                if (normalized.len > sent_structured_len) {
                    const payload = formatChunkJson(&json_buf, normalized[sent_structured_len..], is_chat) catch continue;
                    conn.writeSseEvent(payload) catch {
                        write_failed = true;
                        continue;
                    };
                    sent_structured_len = normalized.len;
                }
            } else {
                const text = ctx.tokenizer.decodeToken(tok, &dec_buf);
                const payload = formatChunkJson(&json_buf, text, is_chat) catch continue;
                conn.writeSseEvent(payload) catch {
                    write_failed = true;
                };
            }
        }
        if (write_failed) ctx.engine.cancel(&chan);
        if (ch.finished) break;
    }
    const generated_tokens = chan.consumed;
    const finish_reason = cudaServeFinishReason(generated_tokens, max_tokens, last_generated_token, ctx.engine.eos, ctx.engine.eot);
    const usage = ServeUsage{ .prompt_tokens = prompt_tokens.len, .completion_tokens = generated_tokens, .total_tokens = prompt_tokens.len + generated_tokens };
    const timings = cudaServeTimings(prompt_tokens.len, generated_tokens, chan.prefill_wall_ns, chan.decode_wall_ns);
    logServeTimings(prompt_tokens.len, generated_tokens, chan.prefill_wall_ns, chan.decode_wall_ns);
    if (generation_failed) {
        log.warn("request {d} failed during GPU generation", .{id});
        const metadata = try buildCudaServeErrorJson(allocator, "GPU generation failed", chan.mtp);
        defer allocator.free(metadata);
        if (stream_response) {
            if (!write_failed) {
                conn.writeSseEvent(metadata) catch {};
                conn.writeSseDone() catch {};
            }
        } else {
            try conn.sendJson(500, metadata);
        }
        return;
    }
    if (stream_response) {
        if (!write_failed) {
            if (!ctx.debug_ids) {
                const metadata = try buildCudaServeFinishJson(allocator, is_chat, finish_reason, chan.mtp, usage, timings);
                defer allocator.free(metadata);
                try conn.writeSseEvent(metadata);
            }
            conn.writeSseDone() catch {};
        }
        return;
    }
    if (write_failed) {
        const metadata = try buildCudaServeErrorJson(allocator, "response buffering failed", chan.mtp);
        defer allocator.free(metadata);
        try conn.sendJson(500, metadata);
        return;
    }

    const response_output = if (structured_chat)
        routes_mod.normalizeStructuredAssistantOutput(ctx.tokenizer, output.items, thinking_enabled, structured_buf.?) catch ""
    else
        output.items;
    const response = try buildCudaServeResponseJson(allocator, is_chat, response_output, finish_reason, chan.mtp, usage, timings);
    defer allocator.free(response);
    try conn.sendJson(200, response);
}

fn logServeTimings(prompt_tokens: usize, generated_tokens: usize, prefill_ns: u64, decode_ns: u64) void {
    const prefill_ms = @as(f64, @floatFromInt(prefill_ns)) / 1_000_000.0;
    const decode_ms = @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0;
    const prefill_tps = if (prefill_ns == 0) 0.0 else @as(f64, @floatFromInt(prompt_tokens)) * 1_000_000_000.0 / @as(f64, @floatFromInt(prefill_ns));
    const decode_tps = if (decode_ns == 0) 0.0 else @as(f64, @floatFromInt(generated_tokens)) * 1_000_000_000.0 / @as(f64, @floatFromInt(decode_ns));
    const ms_per_token = if (generated_tokens == 0) 0.0 else decode_ms / @as(f64, @floatFromInt(generated_tokens));
    inference_timing_log.info("Prefill: {d} tokens in {d:.1} ms ({d:.2} tok/s)", .{ prompt_tokens, prefill_ms, prefill_tps });
    inference_timing_log.info("Generated {d} tokens in {d:.1} ms — {d:.2} tok/s ({d:.1} ms/tok)", .{ generated_tokens, decode_ms, decode_tps, ms_per_token });
}

/// Drain a request's stream without writing (client disconnected) so the worker
/// runs it to completion and frees the slot.
fn drainQuietly(engine: *cuda_serve_mod.ServeEngine, chan: *cuda_serve_mod.ReqChannel) void {
    engine.cancel(chan);
    var buf: [32]u32 = undefined;
    while (true) {
        const ch = engine.nextChunk(chan, &buf);
        if (ch.finished) break;
    }
}

/// Parse a comma-separated list of decimal token ids (gate/debug-ids contract).
fn parseTokenIdList(allocator: std.mem.Allocator, text: []const u8) ![]u32 {
    var list: std.ArrayList(u32) = .{};
    errdefer list.deinit(allocator);
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |s| {
        const t = std.mem.trim(u8, s, " \t\r\n");
        if (t.len == 0) continue;
        try list.append(allocator, try std.fmt.parseInt(u32, t, 10));
    }
    return list.toOwnedSlice(allocator);
}

/// Format one OpenAI-style streaming chunk carrying a single token's text.
fn formatChunkJson(buf: []u8, text: []const u8, is_chat: bool) ![]const u8 {
    var esc: [512]u8 = undefined;
    const escaped = jsonEscape(&esc, text);
    if (is_chat) {
        return std.fmt.bufPrint(buf, "{{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{s}\"}}}}]}}", .{escaped});
    }
    return std.fmt.bufPrint(buf, "{{\"choices\":[{{\"index\":0,\"text\":\"{s}\"}}]}}", .{escaped});
}

/// Minimal JSON string escaper for SSE token payloads (truncates on overflow).
fn jsonEscape(buf: []u8, s: []const u8) []const u8 {
    var n: usize = 0;
    for (s) |c| {
        const rep: []const u8 = switch (c) {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            else => &[_]u8{c},
        };
        if (n + rep.len > buf.len) break;
        @memcpy(buf[n .. n + rep.len], rep);
        n += rep.len;
    }
    return buf[0..n];
}

fn jsonEscapeAlloc(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .{};
    errdefer out.deinit(allocator);
    for (s) |c| {
        const rep: []const u8 = switch (c) {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            else => &[_]u8{c},
        };
        try out.appendSlice(allocator, rep);
    }
    return out.toOwnedSlice(allocator);
}

/// True if an env var is set to an "on" value (1/true/yes/on).
fn envIsOn(name: []const u8) bool {
    const v = std.posix.getenv(name) orelse return false;
    return std.mem.eql(u8, v, "1") or std.ascii.eqlIgnoreCase(v, "true") or
        std.ascii.eqlIgnoreCase(v, "yes") or std.ascii.eqlIgnoreCase(v, "on");
}

// CUDA: `zinc --check` for NVIDIA. Detects the best CUDA device and prints a
// diagnostics report (device name, compute capability, SMs, VRAM, backend),
// then OK. diagnostics_mod is a no-op stub on the CUDA backend, so this is the
// operator-facing preflight on this platform.
fn runCudaCheck(config: Config, check_target: ResolvedCheckTarget, allocator: std.mem.Allocator) !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writerStreaming(&stdout_buffer);
    const w = &stdout_writer.interface;

    try w.print("\n=== ZINC System Diagnostics ===\n", .{});
    try w.print("\n[1/2] Host Environment\n", .{});
    try w.print("  OS: {s} [OK]\n", .{@tagName(builtin.os.tag)});
    try w.print("  CPU arch: {s} [OK]\n", .{@tagName(builtin.cpu.arch)});
    try w.print("  Backend: {s} [OK]\n", .{if (gpu.is_rocm) "rocm" else "cuda"});

    try w.print("\n[2/2] {s} Device\n", .{accelerator_name});
    var device = cuda_device_mod.CudaDevice.initBest(allocator) catch |err| {
        try w.print("  {s} init: FAILED ({s}) [FAIL]\n", .{ accelerator_name, @errorName(err) });
        try w.print("\nVerdict: NOT READY [FAIL]\n", .{});
        try w.flush();
        return error.DiagnosticsFailed;
    };
    defer device.deinit();

    var name_buf: [256]u8 = undefined;
    const name = device.name(&name_buf);
    const cc = device.computeCapability();
    const total = device.totalMemory();
    const free = device.freeMemory();
    try w.print("  {s} init: Initialized best device (index {d}) [OK]\n", .{ accelerator_name, device.device_index });
    try w.print("  Device: {s} [OK]\n", .{name});
    if (comptime gpu.is_rocm) {
        var arch_buf: [64]u8 = undefined;
        try w.print("  Compute target: {s} [OK]\n", .{device.arch(&arch_buf)});
        try w.print("  CU count: {d} [OK]\n", .{device.smCount()});
        try w.print("  Wavefront size: {d}\n", .{device.warpSize()});
    } else {
        try w.print("  Compute capability: sm_{d} [OK]\n", .{cc});
        try w.print("  SM count: {d} [OK]\n", .{device.smCount()});
        try w.print("  Warp size: {d}\n", .{device.warpSize()});
    }
    try w.print("  Total VRAM: {d:.2} GiB [OK]\n", .{bytesToGiBf(total)});
    try w.print("  Free VRAM: {d:.2} GiB [OK]\n", .{bytesToGiBf(free)});

    // Optional model-fit note (catalog size vs free VRAM); GGUF inspection of an
    // installed model when a managed id / path was supplied.
    if (check_target.managed_model) |managed| {
        try w.print("  Managed model: {s} ({s})\n", .{ managed.id, managed.status_label });
        const status = if (managed.size_bytes <= free) "[OK]" else "[WARN]";
        try w.print("  Catalog size: {d:.2} GiB of weights vs {d:.2} GiB free VRAM {s}\n", .{
            bytesToGiBf(managed.size_bytes), bytesToGiBf(free), status,
        });
    }
    if (check_target.model_path) |path| {
        const cfg = loader_cuda_mod.inspectConfig(path, allocator) catch |err| blk: {
            try w.print("  GGUF: inspection failed for {s}: {s} [WARN]\n", .{ path, @errorName(err) });
            break :blk null;
        };
        if (cfg) |c| {
            try w.print("  GGUF: {s} [OK]\n", .{path});
            try w.print("  architecture: {s} | {d} layers | {d} heads ({d} KV) | dim {d} | vocab {d}\n", .{
                @tagName(c.architecture), c.n_layers, c.n_heads, c.n_kv_heads, c.hidden_dim, c.vocab_size,
            });
        }
    }

    _ = config;
    try w.print("\nVerdict: READY [OK]\n", .{});
    try w.flush();
}

/// Bytes → GiB as f64 for diagnostics formatting.
fn bytesToGiBf(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0);
}

/// Parse arguments, dispatch model-management commands, run diagnostics, or start inference.
/// In prompt mode the engine runs up to `max_tokens` forward passes and prints the decoded output.
/// In server mode an HTTP listener is started and requests are handled until SIGINT/SIGTERM.
/// @note Fatal startup errors are logged and the process exits rather than returning an error.
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

    if (config.show_version) {
        var stdout_buffer: [512]u8 = undefined;
        var stdout_writer = std.fs.File.stdout().writerStreaming(&stdout_buffer);
        try build_info.writeVersion(&stdout_writer.interface);
        try stdout_writer.interface.flush();
        return;
    }

    if (config.check) {
        var check_target = resolveCheckTarget(config, allocator) catch |err| {
            log.err("Failed to resolve model for diagnostics: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
        defer check_target.deinit(allocator);

        if (comptime gpu.is_cuda) {
            // CUDA-family backends compile kernels at runtime and do not need a
            // Vulkan/Metal shader directory for their preflight.
            runCudaCheck(config, check_target, allocator) catch |err| {
                log.err("Diagnostics completed with error: {s}", .{@errorName(err)});
                std.process.exit(1);
            };
            return;
        }

        const check_shader_dir_owned = runtime_assets.resolveShaderDir(allocator, if (gpu.is_metal) .metal else .spirv) catch |err| blk: {
            log.warn("Could not resolve shader directory before diagnostics: {s}", .{@errorName(err)});
            break :blk null;
        };
        defer if (check_shader_dir_owned) |shader_dir| allocator.free(shader_dir);
        const check_shader_dir = check_shader_dir_owned orelse if (gpu.is_metal)
            "src/shaders/metal"
        else
            "zig-out/share/zinc/shaders";

        diagnostics_mod.run(.{
            .device_index = config.gpuDevicePreference(),
            .model_path = check_target.model_path,
            .requested_context_length = config.context_length,
            .managed_model = check_target.managed_model,
            .shader_dir = check_shader_dir,
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
        log.debug("Memory: {d} GB | Max buffer: {d} GB", .{
            device.totalMemory() / (1024 * 1024 * 1024),
            device.maxBufferSize() / (1024 * 1024 * 1024),
        });
        log.debug(
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
        log.debug(
            "Inference hints: simdgroup-width comes from pipeline threadExecutionWidth; apple10 => investigate TensorOps/M5 neural accelerators for large GEMMs; unified memory => avoid staging copies; raytracing is irrelevant for inference",
            .{},
        );

        if (config.prompt) |prompt| {
            var gpu_process_lock = process_lock_mod.acquire(.metal, device.selected_device_index) catch |err| {
                reportGpuProcessLockError(err, .metal, device.selected_device_index);
            };
            defer gpu_process_lock.deinit();

            // Load model (zero-copy mmap) for prompt-mode execution.
            var model = metal_loader.load(model_path.?, device.ctx, allocator) catch |err| {
                log.err("Failed to load model: {s}", .{@errorName(err)});
                std.process.exit(1);
            };
            defer model.deinit();
            if (config.context_length) |requested_context| {
                memory_plan.applyRequestedContextLimit(&model.config, requested_context);
            } else {
                const metal_budget = blk: {
                    const working_set = device.recommendedMaxWorkingSetSize();
                    break :blk if (working_set > 0) working_set else device.totalMemory();
                };
                const auto_context = memory_plan.autoContextTokensForDeviceBudget(
                    memory_plan.profile(model.config),
                    metal_loader.residentWeightBytes(&model),
                    metal_budget,
                    model.config.context_length,
                );
                memory_plan.applyRequestedContextLimit(&model.config, auto_context);
            }

            log.info("Prompt: {s}", .{prompt});

            var tokenizer = tokenizer_mod.Tokenizer.initFromGGUF(&model.gguf_file, allocator) catch |err| {
                log.err("Failed to init tokenizer from GGUF: {s}", .{@errorName(err)});
                std.process.exit(1);
            };
            defer tokenizer.deinit();

            const auto_chat = !config.chat and !config.raw_prompt and shouldAutoChatCliPrompt(&tokenizer, prompt);
            const use_chat_prompt = config.chat or auto_chat;
            var prepared_prompt = try prepareCliPrompt(&tokenizer, prompt, use_chat_prompt, config.system_prompt, allocator);
            defer prepared_prompt.deinit(allocator);
            if (use_chat_prompt) {
                log.info("Prompt mode: {s}chat template ({d} chars)", .{
                    if (auto_chat) "auto " else "",
                    prepared_prompt.text.len,
                });
            }

            const prompt_tokens = try tokenizer.encodePrompt(prepared_prompt.text, allocator);
            defer allocator.free(prompt_tokens);

            log.info("Prompt fingerprint: raw={x} prepared={x} mode={s} prompt_tokens={d}", .{
                promptFingerprint(prompt),
                promptFingerprint(prepared_prompt.text),
                if (use_chat_prompt) "chat" else "raw",
                prompt_tokens.len,
            });
            log.info("Prompt tokens ({d}): {any}", .{ prompt_tokens.len, prompt_tokens[0..@min(prompt_tokens.len, 30)] });

            // Initialize inference engine
            var engine = forward_metal.InferenceEngine.init(&model, &device, allocator, .{
                .profile_enabled = config.profile,
                .debug_validation_enabled = config.profile and config.debug,
            }) catch |err| {
                log.err("Failed to init Metal inference engine: {s}", .{@errorName(err)});
                std.process.exit(1);
            };
            defer engine.deinit();

            const eos_id = if (envIsOn("ZINC_BENCH_IGNORE_EOS")) std.math.maxInt(u32) else cliGenerationStopId(&tokenizer, use_chat_prompt);
            if (eos_id == std.math.maxInt(u32)) {
                log.info("Benchmark mode: ignoring EOS until max_tokens", .{});
            }

            // Generate
            const output_tokens = forward_metal.generate(&engine, prompt_tokens, config.max_tokens, eos_id, allocator) catch |err| {
                log.err("Failed to generate: {s}", .{@errorName(err)});
                std.process.exit(1);
            };
            defer allocator.free(output_tokens);

            if (output_tokens.len == 0) {
                log.warn("Metal decode loop not yet implemented. Engine initialized successfully with {d} pipelines.", .{9});
            } else {
                if (config.profile) {
                    const vocab_size = model.config.vocab_size;
                    const logits_src = &engine.logits_buf;
                    const logits_ptr: [*]const f32 = @ptrCast(@alignCast(logits_src.cpu_ptr.?));
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

                if (config.debug) {
                    log.debug("Output tokens ({d}): {any}", .{
                        output_tokens.len,
                        output_tokens[0..@min(output_tokens.len, 20)],
                    });
                    const show_n = @min(output_tokens.len, 5);
                    for (0..show_n) |ti| {
                        const tok_str = if (output_tokens[ti] < tokenizer.vocab.len) tokenizer.vocab[output_tokens[ti]] else "?";
                        log.debug("  gen[{d}]: id={d} \"{s}\"", .{ ti, output_tokens[ti], tok_str });
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
                const output_text = trimCliOutputText(text_buf.items, use_chat_prompt or eos_id == std.math.maxInt(u32));
                log.info("Output ({d} tokens): {s}", .{ output_tokens.len, output_text });
            }
        } else {
            log.info("Server mode — port {d}, max {d} concurrent requests", .{ config.port, config.max_parallel });

            var manager = if (resolved_model) |startup_model|
                model_manager_mod.ModelManager.init(startup_model.spec, &device, allocator) catch |err| {
                    if (err == error.GpuAlreadyReserved) {
                        reportGpuProcessLockError(err, .metal, device.selected_device_index);
                    }
                    log.err("Failed to init Metal model manager: {s}", .{@errorName(err)});
                    std.process.exit(1);
                }
            else
                model_manager_mod.ModelManager.initEmpty(&device, config.context_length, allocator);
            defer manager.deinit();

            runHttpServer(config, &manager, allocator);
        }
        return;
    }

    // CUDA: NVIDIA / WSL2 backend. Placed before the Vulkan tail; the
    // `return` makes the Vulkan code below comptime-unreachable on CUDA (same
    // pattern as the Metal block above), so it is never analyzed here.
    if (comptime gpu.is_cuda) {
        runCuda(config, model_path, allocator) catch |err| {
            log.err("{s} run failed: {s}", .{ accelerator_name, @errorName(err) });
            std.process.exit(1);
        };
        return;
    }

    // Vulkan backend (Linux)
    var vk_instance = instance_mod.Instance.init(allocator, config.gpuDevicePreference()) catch |err| {
        log.err("Vulkan init failed: {s}", .{@errorName(err)});
        std.process.exit(1);
    };
    defer vk_instance.deinit();

    // Detect GPU capabilities
    const gpu_config = gpu_detect.detect(&vk_instance);
    gpu_config.log_info();

    // Determine shader directory — probe relative paths first, then fall back
    // to a path relative to the running executable so that installed layouts
    // (e.g. Nix store: $out/bin/zinc → $out/share/zinc/shaders) work correctly.
    const shader_dir = resolveShaderDir(allocator) catch |err| {
        log.err("Could not locate shader directory: {s}", .{@errorName(err)});
        std.process.exit(1);
    };
    defer allocator.free(shader_dir);

    if (config.prompt) |prompt| {
        var gpu_process_lock = process_lock_mod.acquire(.vulkan, vk_instance.selected_device_index) catch |err| {
            reportGpuProcessLockError(err, .vulkan, vk_instance.selected_device_index);
        };
        defer gpu_process_lock.deinit();

        log.debug("Prompt: {s}", .{prompt});

        var cmd_pool = try CommandPool.init(&vk_instance);
        defer cmd_pool.deinit();

        var model = loader_mod.load(model_path.?, &vk_instance, &cmd_pool, allocator) catch |err| {
            log.err("Failed to load model: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
        defer model.deinit(&vk_instance);
        memory_plan.applyRequestedContextLimit(&model.config, config.context_length);

        var engine = forward_mod.InferenceEngine.init(&model, &vk_instance, gpu_config, shader_dir, allocator) catch |err| {
            log.err("Failed to init inference engine: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
        defer engine.deinit();

        if (wants_graph_artifacts) {
            const graph = engine.decodeGraph() orelse return error.UnsupportedArchitecture;
            writeDecodeGraphArtifacts(graph, config.graph_report_path, config.graph_dot_path, allocator) catch |err| {
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
            if (config.profile) {
                engine.enableValidationDiagnostics();
            }
        }

        // Initialize native BPE tokenizer from GGUF metadata
        var tokenizer = tokenizer_mod.Tokenizer.initFromGGUF(&model.gguf_file, allocator) catch |err| {
            log.err("Failed to init tokenizer from GGUF: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
        defer tokenizer.deinit();

        const auto_chat = !config.chat and !config.raw_prompt and shouldAutoChatCliPrompt(&tokenizer, prompt);
        const use_chat_prompt = config.chat or auto_chat;
        var prepared_prompt = try prepareCliPrompt(&tokenizer, prompt, use_chat_prompt, config.system_prompt, allocator);
        defer prepared_prompt.deinit(allocator);
        if (use_chat_prompt) {
            log.debug("Prompt mode: {s}chat template ({d} chars)", .{
                if (auto_chat) "auto " else "",
                prepared_prompt.text.len,
            });
        }

        // Tokenize prompt into caller-owned storage. This keeps CLI and server
        // prompt construction on the same code path, including BOS handling.
        const prompt_tokens = try tokenizer.encodePrompt(prepared_prompt.text, allocator);
        defer allocator.free(prompt_tokens);

        log.info("Prompt fingerprint: raw={x} prepared={x} mode={s} prompt_tokens={d}", .{
            promptFingerprint(prompt),
            promptFingerprint(prepared_prompt.text),
            if (use_chat_prompt) "chat" else "raw",
            prompt_tokens.len,
        });
        log.info("Prompt tokens ({d}): {any}", .{ prompt_tokens.len, prompt_tokens[0..@min(prompt_tokens.len, 30)] });
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

        const eos_id = if (envIsOn("ZINC_BENCH_IGNORE_EOS")) std.math.maxInt(u32) else cliGenerationStopId(&tokenizer, use_chat_prompt);
        if (eos_id == std.math.maxInt(u32)) {
            log.info("Benchmark mode: ignoring EOS until max_tokens", .{});
        }

        // Generate
        const output_tokens = try forward_mod.generate(&engine, prompt_tokens, config.max_tokens, eos_id, allocator);
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
        // Debug: inspect the last logits, including the legacy Qwen anchors.
        if (config.debug) {
            const logits = engine.lastLogits();
            if (model.config.architecture != .inkling2_loop and logits.len > 11751) {
                log.debug("  logit[11751 'Paris']={d:.4} logit[524 'not']={d:.4} logit[264 'a']={d:.4}", .{
                    logits[11751], logits[524], logits[264],
                });
            }
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
                n_zero, n_nan, n_inf, sum_abs / @as(f64, @floatFromInt(logits.len)),
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
            const structured = use_chat_prompt and tokenizer.detectTemplateKind() == .inkling2;
            const normalized = if (structured) try allocator.alloc(u8, text_buf.items.len) else null;
            defer if (normalized) |buf| allocator.free(buf);
            const output_text = if (normalized) |buf|
                try routes_mod.normalizeStructuredAssistantOutput(&tokenizer, text_buf.items, true, buf)
            else
                trimCliOutputText(text_buf.items, use_chat_prompt or eos_id == std.math.maxInt(u32));
            log.info("Output text: {s}", .{output_text});
            // Also log raw token IDs for debugging
            log.info("Output tokens ({d}): first20={any}", .{
                output_tokens.len,
                output_tokens[0..@min(output_tokens.len, 20)],
            });
        }
    } else {
        log.info("Server mode — port {d}, max {d} concurrent requests", .{ config.port, config.max_parallel });

        var manager = if (resolved_model) |startup_model|
            model_manager_mod.ModelManager.init(startup_model.spec, &vk_instance, gpu_config, shader_dir, allocator) catch |err| {
                if (err == error.GpuAlreadyReserved) {
                    reportGpuProcessLockError(err, .vulkan, vk_instance.selected_device_index);
                }
                log.err("Failed to init model manager: {s}", .{@errorName(err)});
                std.process.exit(1);
            }
        else
            model_manager_mod.ModelManager.initEmpty(&vk_instance, gpu_config, shader_dir, config.context_length, allocator);
        defer manager.deinit();
        runHttpServer(config, &manager, allocator);
    }
}

test "parseArgs: defaults" {
    const args = [_][:0]const u8{"zinc"};
    const config = try parseArgs(&args);
    try std.testing.expectEqual(@as(u16, 8080), config.port);
    try std.testing.expect(config.context_length == null);
    try std.testing.expectEqual(@as(u8, 0), config.kv_quant);
    try std.testing.expect(config.model_path == null);
    try std.testing.expect(config.model_id == null);
    try std.testing.expectEqual(@as(u32, 0), config.device_index);
    try std.testing.expect(!config.device_index_explicit);
    if (comptime gpu.is_vulkan) {
        try std.testing.expectEqual(instance_mod.auto_select_device_index, config.gpuDevicePreference());
    } else {
        try std.testing.expectEqual(@as(u32, 0), config.gpuDevicePreference());
    }
    try std.testing.expect(config.prompt == null);
    try std.testing.expect(config.system_prompt == null);
    try std.testing.expect(!config.chat);
    try std.testing.expect(!config.raw_prompt);
    try std.testing.expectEqual(Command.run, config.command);
}

test "parseArgs: full args" {
    const args = [_][:0]const u8{
        "zinc",         "-m",         "model.gguf",     "--model-id",      "qwen35-9b-q4k-m",
        "-p",           "9090",       "-d",             "1",               "-c",
        "8192",         "--parallel", "8",              "--prompt",        "hello",
        "--max-tokens", "32",         "--chat",         "--system-prompt", "system guidance",
        "--kv-quant",   "3",          "--graph-report", "graph.json",      "--graph-dot",
        "graph.dot",
    };
    const config = try parseArgs(&args);
    try std.testing.expectEqualStrings("model.gguf", config.model_path.?);
    try std.testing.expectEqualStrings("qwen35-9b-q4k-m", config.model_id.?);
    try std.testing.expectEqual(@as(u16, 9090), config.port);
    try std.testing.expectEqual(@as(u32, 1), config.device_index);
    try std.testing.expect(config.device_index_explicit);
    try std.testing.expectEqual(@as(u32, 1), config.gpuDevicePreference());
    try std.testing.expectEqual(@as(?u32, 8192), config.context_length);
    try std.testing.expectEqual(@as(u32, 8), config.max_parallel);
    try std.testing.expectEqualStrings("hello", config.prompt.?);
    try std.testing.expectEqualStrings("system guidance", config.system_prompt.?);
    try std.testing.expectEqual(@as(u32, 32), config.max_tokens);
    try std.testing.expect(config.chat);
    try std.testing.expect(!config.raw_prompt);
    try std.testing.expectEqual(@as(u8, 3), config.kv_quant);
    try std.testing.expectEqualStrings("graph.json", config.graph_report_path.?);
    try std.testing.expectEqualStrings("graph.dot", config.graph_dot_path.?);
}

test "parseArgs: -hf sets hf_spec" {
    const args = [_][:0]const u8{ "zinc", "-hf", "unsloth/Qwen3.5-9B-GGUF:Q4_K_M" };
    const config = try parseArgs(&args);
    try std.testing.expectEqualStrings("unsloth/Qwen3.5-9B-GGUF:Q4_K_M", config.hf_spec.?);
}

test "parseArgs: -hf conflicts with -m and --model-id" {
    const with_model = [_][:0]const u8{ "zinc", "-hf", "a/b", "-m", "model.gguf" };
    try std.testing.expectError(error.ConflictingModelSources, parseArgs(&with_model));
    const with_id = [_][:0]const u8{ "zinc", "-hf", "a/b", "--model-id", "qwen35-9b-q4k-m" };
    try std.testing.expectError(error.ConflictingModelSources, parseArgs(&with_id));
}

test "CUDA serving clamps Gemma MoE to one slot" {
    try std.testing.expectEqual(@as(u32, 1), cudaServeSlotCount(4, true));
    try std.testing.expectEqual(@as(u32, 8), cudaServeSlotCount(8, false));
    try std.testing.expectEqual(@as(u32, 1), cudaServeSlotCount(0, false));
    try std.testing.expectEqual(@as(u32, 64), cudaServeSlotCount(128, false));
}

test "parseArgs: allows large context requests" {
    const args = [_][:0]const u8{ "zinc", "-c", "65536" };
    const config = try parseArgs(&args);
    try std.testing.expectEqual(@as(?u32, 65536), config.context_length);
}

test "parseArgs: -c 0 requests auto-sizing" {
    const args = [_][:0]const u8{ "zinc", "-c", "0" };
    const config = try parseArgs(&args);
    try std.testing.expect(config.context_length == null);
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

test "parseArgs: version flag" {
    const args = [_][:0]const u8{ "zinc", "--version" };
    const config = try parseArgs(&args);
    try std.testing.expect(config.show_version);
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

test "parseArgs: raw flag" {
    const args = [_][:0]const u8{ "zinc", "--prompt", "hi", "--raw" };
    const config = try parseArgs(&args);
    try std.testing.expect(config.raw_prompt);
    try std.testing.expect(!config.chat);
    try std.testing.expectEqualStrings("hi", config.prompt.?);
}

test "parseArgs: raw and chat conflict" {
    const args = [_][:0]const u8{ "zinc", "--prompt", "hi", "--chat", "--raw" };
    try std.testing.expectError(error.ConflictingPromptModes, parseArgs(&args));
}

test "parseArgs: system prompt requires explicit CLI chat mode" {
    const valid_args = [_][:0]const u8{ "zinc", "--prompt", "hi", "--chat", "--system-prompt", "be direct" };
    const valid = try parseArgs(&valid_args);
    try std.testing.expectEqualStrings("be direct", valid.system_prompt.?);

    const no_chat = [_][:0]const u8{ "zinc", "--prompt", "hi", "--system-prompt", "be direct" };
    try std.testing.expectError(error.SystemPromptRequiresChat, parseArgs(&no_chat));

    const no_prompt = [_][:0]const u8{ "zinc", "--chat", "--system-prompt", "be direct" };
    try std.testing.expectError(error.SystemPromptRequiresChat, parseArgs(&no_prompt));
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

    const pull_args = [_][:0]const u8{ "zinc", "model", "pull", "qwen35-9b-q4k-m" };
    const pull_config = try parseArgs(&pull_args);
    try std.testing.expectEqual(Command.model_pull, pull_config.command);
    try std.testing.expectEqualStrings("qwen35-9b-q4k-m", pull_config.command_model_id.?);

    const active_args = [_][:0]const u8{ "zinc", "model", "active" };
    const active_config = try parseArgs(&active_args);
    try std.testing.expectEqual(Command.model_active, active_config.command);

    const rm_args = [_][:0]const u8{ "zinc", "model", "rm", "-f", "qwen35-9b-q4k-m" };
    const rm_config = try parseArgs(&rm_args);
    try std.testing.expectEqual(Command.model_rm, rm_config.command);
    try std.testing.expect(rm_config.command_force);
    try std.testing.expectEqualStrings("qwen35-9b-q4k-m", rm_config.command_model_id.?);

    const use_args = [_][:0]const u8{ "zinc", "model", "use", "qwen35-9b-q4k-m", "--force" };
    const use_config = try parseArgs(&use_args);
    try std.testing.expectEqual(Command.model_use, use_config.command);
    try std.testing.expect(use_config.command_force);
    try std.testing.expectEqualStrings("qwen35-9b-q4k-m", use_config.command_model_id.?);
}

test "parseArgs: chat command" {
    const args = [_][:0]const u8{ "zinc", "chat", "--model-id", "qwen35-9b-q4k-m" };
    const config = try parseArgs(&args);
    try std.testing.expectEqual(Command.chat, config.command);
    try std.testing.expectEqualStrings("qwen35-9b-q4k-m", config.model_id.?);
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
        .model_id = "qwen35-9b-q4k-m",
        .model_path = "raw.gguf",
    };
    var target = try resolveCheckTarget(config, std.testing.allocator);
    defer target.deinit(std.testing.allocator);

    try std.testing.expect(target.managed_model != null);
    try std.testing.expectEqualStrings("qwen35-9b-q4k-m", target.managed_model.?.id);
    if (target.model_path) |path| {
        try std.testing.expect(!std.mem.eql(u8, path, "raw.gguf"));
    }
}

/// Locate the compiled SPIR-V shader directory. Returns an allocated slice
/// owned by the caller.
fn resolveShaderDir(allocator: std.mem.Allocator) ![]u8 {
    return runtime_assets.resolveShaderDir(allocator, .spirv);
}

/// Test-friendly variant: probe `base_dir` for the cwd-relative candidates,
/// then fall back to `exe_dir_override/../share/zinc/shaders` (or the
/// running executable's directory when override is null).
fn resolveShaderDirFrom(allocator: std.mem.Allocator, base_dir: std.fs.Dir, exe_dir_override: ?[]const u8) ![]u8 {
    return runtime_assets.resolveShaderDirFrom(allocator, base_dir, exe_dir_override, .spirv);
}

test "resolveShaderDirFrom finds first cwd-relative candidate" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.makePath("zig-out/share/zinc/shaders");
    const result = try resolveShaderDirFrom(std.testing.allocator, tmp.dir, "/nonexistent");
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("zig-out/share/zinc/shaders", result);
}

test "resolveShaderDirFrom falls back to second candidate when first missing" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.makePath("share/zinc/shaders");
    const result = try resolveShaderDirFrom(std.testing.allocator, tmp.dir, "/nonexistent");
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("share/zinc/shaders", result);
}

test "resolveShaderDirFrom falls back to exe-relative when cwd has nothing" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // Build an installed layout: <prefix>/bin/zinc with shaders at
    // <prefix>/share/zinc/shaders. POSIX path resolution requires each
    // component on the way to ".." to exist as a real directory, so create
    // both bin/ and share/zinc/shaders/.
    try tmp.dir.makePath("install/bin");
    try tmp.dir.makePath("install/share/zinc/shaders");
    const bin_dir = try tmp.dir.realpathAlloc(std.testing.allocator, "install/bin");
    defer std.testing.allocator.free(bin_dir);

    var empty = std.testing.tmpDir(.{});
    defer empty.cleanup();
    const result = try resolveShaderDirFrom(std.testing.allocator, empty.dir, bin_dir);
    defer std.testing.allocator.free(result);
    // The derived path is bin_dir + ../share/zinc/shaders → install/share/zinc/shaders.
    try std.testing.expect(std.mem.endsWith(u8, result, "share/zinc/shaders"));
}

test "resolveShaderDirFrom returns ShaderDirNotFound when no path exists" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try std.testing.expectError(error.ShaderDirNotFound, resolveShaderDirFrom(std.testing.allocator, tmp.dir, "/this/path/has/no/shaders"));
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

    var prepared = try prepareCliPrompt(&tok, "hello", false, null, std.testing.allocator);
    defer prepared.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("hello", prepared.text);
    try std.testing.expect(prepared.owned_buf == null);
}

test "prepareCliPrompt returns full owned chat buffer" {
    var tok = makeTestTokenizer(null);
    defer tok.token_to_id.deinit();

    var prepared = try prepareCliPrompt(&tok, "Hello", true, null, std.testing.allocator);
    defer prepared.deinit(std.testing.allocator);

    try std.testing.expect(prepared.owned_buf != null);
    try std.testing.expect(prepared.text.ptr == prepared.owned_buf.?.ptr);
    try std.testing.expect(prepared.owned_buf.?.len >= prepared.text.len);
    try std.testing.expectEqualStrings("<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n", prepared.text);
}

test "prepareCliPrompt uses closed think scaffold for qwen chatml templates" {
    var tok = makeTestTokenizer(
        \\{%- if add_generation_prompt %}
        \\  {{- '<|im_start|>assistant\n' }}
        \\  {%- if enable_thinking is defined and enable_thinking is true %}
        \\    {{- '<think>\n' }}
        \\  {%- else %}
        \\    {{- '<think>\n\n</think>\n\n' }}
        \\  {%- endif %}
        \\{%- endif %}
    );
    defer tok.token_to_id.deinit();

    var prepared = try prepareCliPrompt(&tok, "Hello", true, null, std.testing.allocator);
    defer prepared.deinit(std.testing.allocator);

    try std.testing.expect(std.mem.indexOf(u8, prepared.text, "<|im_start|>user\nHello<|im_end|>\n") != null);
    try std.testing.expect(std.mem.endsWith(u8, prepared.text, "<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}

test "prepareCliPrompt prepends an explicit system turn" {
    var tok = makeTestTokenizer(null);
    defer tok.token_to_id.deinit();

    var prepared = try prepareCliPrompt(
        &tok,
        "Review this code.",
        true,
        "You are a helpful assistant. Answer directly. Do not show analysis.",
        std.testing.allocator,
    );
    defer prepared.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings(
        "<|im_start|>system\nYou are a helpful assistant. Answer directly. Do not show analysis.<|im_end|>\n" ++
            "<|im_start|>user\nReview this code.<|im_end|>\n<|im_start|>assistant\n",
        prepared.text,
    );
}

test "prepareCliPrompt uses gemma4 default closed-thought prompt in chat mode" {
    var tok = makeTestTokenizer(
        "{%- if enable_thinking is defined and enable_thinking -%}<|turn>system\n<|think|><turn|>\n{%- endif -%}<|turn>{{ role }}\n{{ content }}<turn|>\n{%- if add_generation_prompt -%}<|turn>model\n<|channel>thought\n<channel|>{%- endif -%}",
    );
    defer tok.token_to_id.deinit();
    tok.prepend_bos = true;
    tok.bos_id = 2;

    var prepared = try prepareCliPrompt(&tok, "Hello", true, null, std.testing.allocator);
    defer prepared.deinit(std.testing.allocator);

    try std.testing.expect(std.mem.indexOf(u8, prepared.text, "<bos><|turn>system\n<|think|><turn|>\n") == null);
    try std.testing.expect(std.mem.indexOf(u8, prepared.text, "<bos><|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>") != null);
}

test "requestedIncludedWord extracts explicit CLI include-word constraint" {
    try std.testing.expectEqualStrings("benchmark", requestedIncludedWord("Focus on kernels and include the word benchmark.").?);
    try std.testing.expectEqualStrings("Benchmark", requestedIncludedWord("Include the word \"Benchmark\" in the first sentence.").?);
    try std.testing.expect(requestedIncludedWord("Focus on kernels.") == null);
}

test "prepareCliPrompt frontloads gemma4 include-word constraint in user turn" {
    var tok = makeTestTokenizer(
        "{%- if enable_thinking is defined and enable_thinking -%}<|turn>system\n<|think|><turn|>\n{%- endif -%}<|turn>{{ role }}\n{{ content }}<turn|>\n{%- if add_generation_prompt -%}<|turn>model\n<|channel>thought\n<channel|>{%- endif -%}",
    );
    defer tok.token_to_id.deinit();
    tok.prepend_bos = true;
    tok.bos_id = 2;

    var prepared = try prepareCliPrompt(
        &tok,
        "Write an implementation plan. Include the word benchmark.",
        true,
        null,
        std.testing.allocator,
    );
    defer prepared.deinit(std.testing.allocator);

    try std.testing.expect(std.mem.indexOf(u8, prepared.text, "<|turn>system\n") == null);
    try std.testing.expect(std.mem.indexOf(u8, prepared.text, "<|turn>user\nBegin the answer with the exact word benchmark. That word must be the first output word, then continue with the requested answer.\n\nWrite an implementation plan. Include the word benchmark.<turn|>\n") != null);
}

test "shouldAutoChatCliPrompt enables gemma4 turn templates" {
    var tok = makeTestTokenizer(
        "{%- if add_generation_prompt -%}<|turn>model\n<|channel>thought\n<channel|>{%- endif -%}",
    );
    defer tok.token_to_id.deinit();

    try std.testing.expect(shouldAutoChatCliPrompt(&tok, "Hello"));
    try std.testing.expect(!shouldAutoChatCliPrompt(&tok, "<|turn>user\nHello<turn|>"));
}

test "Muse Glimmer CLI auto-chat uses ATEM end-of-turn token" {
    var tok = makeTestTokenizer(
        "<atem:tool_call><|start|>{{ role }}<|message|>{{ content }}<|eot|>",
    );
    defer tok.token_to_id.deinit();
    tok.eot_id = 200021;

    try std.testing.expect(shouldAutoChatCliPrompt(&tok, "Hello"));
    try std.testing.expect(!shouldAutoChatCliPrompt(&tok, "<|start|>user<|message|>Hello<|eot|>"));
    try std.testing.expectEqual(@as(u32, 200021), cliGenerationStopId(&tok, true));
    try std.testing.expectEqual(tok.eosId(), cliGenerationStopId(&tok, false));
}

test "shouldAutoChatCliPrompt leaves non-gemma templates raw" {
    var tok = makeTestTokenizer("<|im_start|>assistant\n");
    defer tok.token_to_id.deinit();

    try std.testing.expect(!shouldAutoChatCliPrompt(&tok, "Hello"));
}

test "trimCliOutputText strips chat terminator only in chat mode" {
    try std.testing.expectEqualStrings("Paris", trimCliOutputText("Paris<|im_end|>", true));
    try std.testing.expectEqualStrings("Paris", trimCliOutputText("Paris<turn|>", true));
    try std.testing.expectEqualStrings("Paris", trimCliOutputText("Paris<end_of_turn>", true));
    try std.testing.expectEqualStrings("Paris<|im_end|>", trimCliOutputText("Paris<|im_end|>", false));
    try std.testing.expectEqualStrings("Paris", trimCliOutputText("Paris", true));
}

test "trimCliOutputText extracts GPT-OSS Harmony final channel" {
    const raw =
        "<|channel|>analysis<|message|>Need answer.<|end|>" ++
        "<|start|>assistant<|channel|>final<|message|> Paris <|return|>";

    try std.testing.expectEqualStrings("Paris", trimCliOutputText(raw, true));
    try std.testing.expectEqualStrings(raw, trimCliOutputText(raw, false));
}
