//! Route dispatcher and endpoint handlers for the OpenAI-compatible API.
//! @section API Server
//! Handles /v1/chat/completions, /v1/completions, /v1/models, /health,
//! and a built-in chat UI. Supports both streaming (SSE) and non-streaming responses.
const std = @import("std");
const http = @import("http.zig");
const forward_mod = @import("../compute/forward.zig");
const catalog_mod = @import("../model/catalog.zig");
const managed_mod = @import("../model/managed.zig");
const model_manager_mod = @import("model_manager.zig");
const tokenizer_mod = @import("../model/tokenizer.zig");
const Model = @import("../model/loader.zig").Model;

const log = std.log.scoped(.routes);

pub const ServerState = struct {
    started_at: i64,
    active_requests: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    queued_requests: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    generation_mutex: std.Thread.Mutex = .{},
    downloads: DownloadTracker = .{},

    pub fn init(started_at: i64) ServerState {
        return .{ .started_at = started_at };
    }

    pub fn uptimeSeconds(self: *const ServerState, now: i64) u64 {
        return @intCast(@max(now - self.started_at, 0));
    }

    pub fn snapshot(self: *const ServerState, now: i64) HealthSnapshot {
        return .{
            .active_requests = self.active_requests.load(.monotonic),
            .queued_requests = self.queued_requests.load(.monotonic),
            .uptime_seconds = self.uptimeSeconds(now),
        };
    }
};

const HealthSnapshot = struct {
    active_requests: u32,
    queued_requests: u32,
    uptime_seconds: u64,
};

const DownloadPhase = enum {
    idle,
    downloading,
    verifying,
    failed,
};

const DownloadSnapshot = struct {
    active: bool,
    phase: DownloadPhase,
    model_id_len: usize,
    model_id_buf: [96]u8,
    downloaded_bytes: u64,
    total_bytes: u64,
    error_len: usize,
    error_buf: [160]u8,

    fn modelId(self: *const DownloadSnapshot) []const u8 {
        return self.model_id_buf[0..self.model_id_len];
    }

    fn errorMessage(self: *const DownloadSnapshot) []const u8 {
        return self.error_buf[0..self.error_len];
    }
};

const DownloadTracker = struct {
    mutex: std.Thread.Mutex = .{},
    active: bool = false,
    phase: DownloadPhase = .idle,
    model_id_len: usize = 0,
    model_id_buf: [96]u8 = [_]u8{0} ** 96,
    downloaded_bytes: u64 = 0,
    total_bytes: u64 = 0,
    error_len: usize = 0,
    error_buf: [160]u8 = [_]u8{0} ** 160,

    fn snapshot(self: *DownloadTracker) DownloadSnapshot {
        self.mutex.lock();
        defer self.mutex.unlock();
        return .{
            .active = self.active,
            .phase = self.phase,
            .model_id_len = self.model_id_len,
            .model_id_buf = self.model_id_buf,
            .downloaded_bytes = self.downloaded_bytes,
            .total_bytes = self.total_bytes,
            .error_len = self.error_len,
            .error_buf = self.error_buf,
        };
    }

    fn begin(self: *DownloadTracker, model_id: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.active) return error.DownloadInProgress;
        if (model_id.len > self.model_id_buf.len) return error.ModelIdTooLong;
        self.active = true;
        self.phase = .downloading;
        self.model_id_len = model_id.len;
        @memcpy(self.model_id_buf[0..model_id.len], model_id);
        self.downloaded_bytes = 0;
        self.total_bytes = 0;
        self.error_len = 0;
    }

    fn updateStart(self: *DownloadTracker, total_bytes: ?u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.phase = .downloading;
        self.total_bytes = total_bytes orelse 0;
    }

    fn updateProgress(self: *DownloadTracker, downloaded_bytes: u64, total_bytes: ?u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.phase = .downloading;
        self.downloaded_bytes = downloaded_bytes;
        self.total_bytes = total_bytes orelse self.total_bytes;
    }

    fn markVerifying(self: *DownloadTracker, downloaded_bytes: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.phase = .verifying;
        self.downloaded_bytes = downloaded_bytes;
        if (self.total_bytes == 0) self.total_bytes = downloaded_bytes;
    }

    fn markComplete(self: *DownloadTracker, downloaded_bytes: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.active = false;
        self.phase = .idle;
        self.downloaded_bytes = downloaded_bytes;
        self.total_bytes = downloaded_bytes;
        self.error_len = 0;
    }

    fn markFailed(self: *DownloadTracker, message: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.active = false;
        self.phase = .failed;
        self.error_len = @min(message.len, self.error_buf.len);
        @memcpy(self.error_buf[0..self.error_len], message[0..self.error_len]);
    }
};

const GenerationGuard = struct {
    state: *ServerState,

    fn acquire(state: *ServerState) GenerationGuard {
        _ = state.queued_requests.fetchAdd(1, .monotonic);
        state.generation_mutex.lock();
        _ = state.queued_requests.fetchSub(1, .monotonic);
        _ = state.active_requests.fetchAdd(1, .monotonic);
        return .{ .state = state };
    }

    fn release(self: *GenerationGuard) void {
        _ = self.state.active_requests.fetchSub(1, .monotonic);
        self.state.generation_mutex.unlock();
    }
};

/// Handle one HTTP connection: parse request, dispatch to endpoint, send response.
/// @param conn Active client connection to read from and write to.
/// @param engine Inference engine for running generation.
/// @param tokenizer Tokenizer for prompt encoding and token decoding.
/// @param model Loaded model (used for model name in API responses).
/// @param server_state Shared server metrics and generation lock.
/// @param allocator Allocator for per-request temporaries.
pub fn handleConnection(
    conn: *http.Connection,
    manager: *model_manager_mod.ModelManager,
    server_state: *ServerState,
    allocator: std.mem.Allocator,
) !void {
    const request = conn.readRequest() catch |err| {
        log.warn("Failed to parse request: {s}", .{@errorName(err)});
        conn.sendError(400, "invalid_request_error", "Malformed HTTP request") catch {};
        return;
    };

    log.info("{s} {s}", .{ @tagName(request.method), request.path });

    // Route dispatch
    if (request.method == .GET and std.mem.eql(u8, request.path, "/health")) {
        try handleHealth(conn, manager, server_state);
    } else if (request.method == .GET and std.mem.eql(u8, request.path, "/v1/models")) {
        try handleModels(conn, manager, server_state, allocator);
    } else if (request.method == .POST and std.mem.eql(u8, request.path, "/v1/models/activate")) {
        try handleActivateModel(conn, manager, server_state, request.body, allocator);
    } else if (request.method == .POST and std.mem.eql(u8, request.path, "/v1/models/pull")) {
        try handlePullModel(conn, manager, server_state, request.body, allocator);
    } else if (request.method == .POST and std.mem.eql(u8, request.path, "/v1/chat/completions")) {
        try handleChatCompletions(conn, manager, server_state, request.body, allocator);
    } else if (request.method == .POST and std.mem.eql(u8, request.path, "/v1/completions")) {
        try handleCompletions(conn, manager, server_state, request.body, allocator);
    } else if (request.method == .OPTIONS) {
        // CORS preflight
        try conn.sendJson(200, "{}");
    } else if (request.method == .GET and (std.mem.eql(u8, request.path, "/") or std.mem.eql(u8, request.path, "/chat"))) {
        try serveChatUi(conn);
    } else {
        try conn.sendError(404, "not_found", "Unknown endpoint");
    }
}

// ── /health ──────────────────────────────────────────────────

fn buildHealthJson(
    server_state: *const ServerState,
    model_name: []const u8,
    memory_usage: model_manager_mod.ModelManager.MemoryUsage,
    buf: []u8,
) ![]const u8 {
    const now = std.time.timestamp();
    const snapshot = server_state.snapshot(now);
    return std.fmt.bufPrint(buf,
        \\{{"status":"ok","model":"{s}","active_requests":{d},"queued_requests":{d},"uptime_seconds":{d},"gpu_memory_used_bytes":{d},"gpu_memory_budget_bytes":{d}}}
    , .{
        model_name,
        snapshot.active_requests,
        snapshot.queued_requests,
        snapshot.uptime_seconds,
        memory_usage.device_local_bytes,
        memory_usage.device_local_budget_bytes,
    });
}

fn handleHealth(conn: *http.Connection, manager: *model_manager_mod.ModelManager, server_state: *const ServerState) !void {
    var buf: [1024]u8 = undefined;
    const body = buildHealthJson(server_state, manager.activeDisplayName(), manager.currentMemoryUsage(), &buf) catch return error.BufferTooSmall;
    try conn.sendJson(200, body);
}

// ── /v1/models ───────────────────────────────────────────────

fn handleModels(
    conn: *http.Connection,
    manager: *model_manager_mod.ModelManager,
    server_state: *ServerState,
    allocator: std.mem.Allocator,
) !void {
    var view = try manager.collectCatalogView(allocator, false);
    defer view.deinit(allocator);
    const memory_usage = manager.currentMemoryUsage();
    const download = server_state.downloads.snapshot();

    var body: std.ArrayList(u8) = .{};
    defer body.deinit(allocator);

    try body.writer(allocator).print(
        "{{\"object\":\"list\",\"profile\":\"{s}\",\"active_memory_used_bytes\":{d},\"active_memory_budget_bytes\":{d},\"data\":[",
        .{ view.profile, memory_usage.device_local_bytes, memory_usage.device_local_budget_bytes },
    );
    const ts = @divTrunc(std.time.timestamp(), 1);
    for (view.data, 0..) |entry, i| {
        if (i != 0) try body.append(allocator, ',');
        const fit_source = if (entry.exact_fit) "exact" else "catalog";
        const is_download_target = download.model_id_len != 0 and std.mem.eql(u8, download.modelId(), entry.id);
        const downloading = is_download_target and download.active;
        const download_phase = if (is_download_target) @tagName(download.phase) else "idle";
        const download_error = if (is_download_target) download.errorMessage() else "";
        try body.writer(allocator).print(
            \\{{"id":"{s}","object":"model","created":{d},"owned_by":"zinc","display_name":"{s}","homepage_url":"{s}","installed":{s},"active":{s},"managed":{s},"supported_on_current_gpu":{s},"fits_current_gpu":{s},"required_vram_bytes":{d},"fit_source":"{s}","status":"{s}","downloading":{s},"download_phase":"{s}","downloaded_bytes":{d},"download_total_bytes":{d},"download_error":"{s}"}} 
        , .{
            entry.id,
            ts,
            entry.display_name,
            entry.homepage_url,
            if (entry.installed) "true" else "false",
            if (entry.active) "true" else "false",
            if (entry.managed) "true" else "false",
            if (entry.supported_on_current_gpu) "true" else "false",
            if (entry.fits_current_gpu) "true" else "false",
            entry.required_vram_bytes,
            fit_source,
            entry.status_label,
            if (downloading) "true" else "false",
            download_phase,
            if (is_download_target) download.downloaded_bytes else 0,
            if (is_download_target) download.total_bytes else 0,
            download_error,
        });
    }
    try body.appendSlice(allocator, "]}");

    try conn.sendJson(200, body.items);
}

fn handleActivateModel(
    conn: *http.Connection,
    manager: *model_manager_mod.ModelManager,
    server_state: *ServerState,
    body: []const u8,
    allocator: std.mem.Allocator,
) !void {
    var parsed = parseRequestBody(body, allocator) catch {
        try conn.sendError(400, "invalid_request_error", "Invalid JSON in request body");
        return;
    };
    defer parsed.deinit();
    if (parsed.model_id.len == 0) {
        try conn.sendError(400, "invalid_request_error", "Field 'model' is required");
        return;
    }

    server_state.generation_mutex.lock();
    defer server_state.generation_mutex.unlock();

    manager.activateManagedModel(parsed.model_id, true) catch |err| {
        const msg = switch (err) {
            error.UnknownManagedModel => "Unknown managed model id",
            error.ModelNotInstalled => "Model is not installed in the local cache",
            error.ModelUnsupportedOnThisGpu => "Model is not marked supported for the current GPU profile",
            error.ModelDoesNotFit => "Model does not fit the current GPU memory budget",
            else => "Model activation failed",
        };
        try conn.sendError(400, "invalid_request_error", msg);
        return;
    };

    var buf: [512]u8 = undefined;
    const response = std.fmt.bufPrint(&buf,
        \\{{"object":"model.activation","id":"{s}","active":true}}
    , .{parsed.model_id}) catch return error.BufferTooSmall;
    try conn.sendJson(200, response);
}

const NullLogWriter = struct {
    pub fn print(self: *NullLogWriter, comptime fmt: []const u8, args: anytype) !void {
        _ = self;
        _ = fmt;
        _ = args;
    }

    pub fn writeAll(self: *NullLogWriter, bytes: []const u8) !void {
        _ = self;
        _ = bytes;
    }

    pub fn flush(self: *NullLogWriter) !void {
        _ = self;
    }
};

const DownloadWorker = struct {
    entry: catalog_mod.CatalogEntry,
    tracker: *DownloadTracker,

    fn run(self: *DownloadWorker) void {
        defer std.heap.page_allocator.destroy(self);

        var sink = NullLogWriter{};
        const observer = managed_mod.DownloadObserver{
            .context = self.tracker,
            .on_start = downloadObserverStart,
            .on_progress = downloadObserverProgress,
            .on_verifying = downloadObserverVerifying,
            .on_complete = downloadObserverComplete,
        };

        managed_mod.pullModelWithObserver(self.entry, std.heap.page_allocator, &sink, &observer) catch |err| {
            self.tracker.markFailed(@errorName(err));
            return;
        };
    }
};

fn downloadObserverStart(context: ?*anyopaque, total_bytes: ?u64) void {
    const tracker: *DownloadTracker = @ptrCast(@alignCast(context.?));
    tracker.updateStart(total_bytes);
}

fn downloadObserverProgress(context: ?*anyopaque, downloaded_bytes: u64, total_bytes: ?u64) void {
    const tracker: *DownloadTracker = @ptrCast(@alignCast(context.?));
    tracker.updateProgress(downloaded_bytes, total_bytes);
}

fn downloadObserverVerifying(context: ?*anyopaque, downloaded_bytes: u64) void {
    const tracker: *DownloadTracker = @ptrCast(@alignCast(context.?));
    tracker.markVerifying(downloaded_bytes);
}

fn downloadObserverComplete(context: ?*anyopaque, downloaded_bytes: u64) void {
    const tracker: *DownloadTracker = @ptrCast(@alignCast(context.?));
    tracker.markComplete(downloaded_bytes);
}

fn handlePullModel(
    conn: *http.Connection,
    manager: *model_manager_mod.ModelManager,
    server_state: *ServerState,
    body: []const u8,
    allocator: std.mem.Allocator,
) !void {
    var parsed = parseRequestBody(body, allocator) catch {
        try conn.sendError(400, "invalid_request_error", "Invalid JSON in request body");
        return;
    };
    defer parsed.deinit();
    if (parsed.model_id.len == 0) {
        try conn.sendError(400, "invalid_request_error", "Field 'model' is required");
        return;
    }

    const entry = catalog_mod.find(parsed.model_id) orelse {
        try conn.sendError(400, "invalid_request_error", "Unknown managed model id");
        return;
    };

    const profile = catalog_mod.profileForGpu(manager.gpu_config);
    if (!catalog_mod.supportedOnCurrentGpu(entry.*, profile, manager.vram_budget_bytes)) {
        try conn.sendError(400, "invalid_request_error", "Model is not marked supported for the current GPU profile or VRAM budget");
        return;
    }

    if (managed_mod.isInstalled(parsed.model_id, allocator)) {
        var installed_buf: [512]u8 = undefined;
        const installed_response = std.fmt.bufPrint(&installed_buf,
            \\{{"object":"model.pull","id":"{s}","state":"installed"}}
        , .{parsed.model_id}) catch return error.BufferTooSmall;
        try conn.sendJson(200, installed_response);
        return;
    }

    server_state.downloads.begin(parsed.model_id) catch {
        const snapshot = server_state.downloads.snapshot();
        if (snapshot.model_id_len != 0 and std.mem.eql(u8, snapshot.modelId(), parsed.model_id)) {
            var busy_buf: [768]u8 = undefined;
            const busy_response = std.fmt.bufPrint(&busy_buf,
                \\{{"object":"model.pull","id":"{s}","state":"{s}","downloaded_bytes":{d},"download_total_bytes":{d}}}
            , .{ parsed.model_id, @tagName(snapshot.phase), snapshot.downloaded_bytes, snapshot.total_bytes }) catch return error.BufferTooSmall;
            try conn.sendJson(202, busy_response);
            return;
        }
        try conn.sendError(409, "invalid_request_error", "Another model download is already in progress");
        return;
    };

    const worker = try std.heap.page_allocator.create(DownloadWorker);
    worker.* = .{
        .entry = entry.*,
        .tracker = &server_state.downloads,
    };

    const thread = std.Thread.spawn(.{}, DownloadWorker.run, .{worker}) catch |err| {
        server_state.downloads.markFailed(@errorName(err));
        std.heap.page_allocator.destroy(worker);
        return err;
    };
    thread.detach();

    var buf: [768]u8 = undefined;
    const response = std.fmt.bufPrint(&buf,
        \\{{"object":"model.pull","id":"{s}","state":"downloading","downloaded_bytes":0,"download_total_bytes":0}}
    , .{parsed.model_id}) catch return error.BufferTooSmall;
    try conn.sendJson(202, response);
}

// ── /v1/chat/completions ─────────────────────────────────────

const chat_stop_strs = [_][]const u8{
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
};

const utf8_replacement = "\xEF\xBF\xBD";
fn findFirstStop(text: []const u8, stop_strs: []const []const u8) ?usize {
    var first: ?usize = null;
    for (stop_strs) |stop| {
        if (std.mem.indexOf(u8, text, stop)) |pos| {
            if (first == null or pos < first.?) {
                first = pos;
            }
        }
    }
    return first;
}

fn trimTrailingChatArtifacts(text: []const u8) []const u8 {
    var out = text;
    while (true) {
        const trimmed = std.mem.trimRight(u8, out, " \t\r\n");
        if (std.mem.endsWith(u8, trimmed, "<|endoftext|>")) {
            out = trimmed[0 .. trimmed.len - "<|endoftext|>".len];
            continue;
        }
        if (std.mem.endsWith(u8, trimmed, utf8_replacement)) {
            out = trimmed[0 .. trimmed.len - utf8_replacement.len];
            continue;
        }
        out = trimmed;
        break;
    }
    const trimmed = out;
    if (trimmed.len > 0 and trimmed[trimmed.len - 1] == '"') {
        var quote_count: usize = 0;
        for (trimmed) |c| {
            if (c == '"') quote_count += 1;
        }
        const prev = if (trimmed.len >= 2) trimmed[trimmed.len - 2] else 0;
        if (quote_count == 1 and (prev == '.' or prev == '!' or prev == '?' or prev == ',' or prev == ':' or prev == ';')) {
            return std.mem.trimRight(u8, trimmed[0 .. trimmed.len - 1], " \t\r\n");
        }
    }
    return trimmed;
}

fn hasDanglingTrailingQuote(text: []const u8) bool {
    const trimmed = std.mem.trimRight(u8, text, " \t\r\n");
    if (trimmed.len == 0 or trimmed[trimmed.len - 1] != '"') return false;
    var quote_count: usize = 0;
    for (trimmed) |c| {
        if (c == '"') quote_count += 1;
    }
    if (quote_count != 1) return false;
    const prev = if (trimmed.len >= 2) trimmed[trimmed.len - 2] else 0;
    return prev == '.' or prev == '!' or prev == '?' or prev == ',' or prev == ':' or prev == ';';
}

fn isReplacementArtifact(text: []const u8) bool {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0 or trimmed.len % utf8_replacement.len != 0) return false;
    var i: usize = 0;
    while (i < trimmed.len) : (i += utf8_replacement.len) {
        if (!std.mem.eql(u8, trimmed[i .. i + utf8_replacement.len], utf8_replacement)) return false;
    }
    return true;
}

fn handleChatCompletions(
    conn: *http.Connection,
    manager: *model_manager_mod.ModelManager,
    server_state: *ServerState,
    body: []const u8,
    allocator: std.mem.Allocator,
) !void {
    var parsed = parseRequestBody(body, allocator) catch {
        try conn.sendError(400, "invalid_request_error", "Invalid JSON in request body");
        return;
    };
    defer parsed.deinit();

    const messages = parsed.messages;
    if (messages.len == 0) {
        try conn.sendError(400, "invalid_request_error", "Field 'messages' is required");
        return;
    }

    var generation_guard = GenerationGuard.acquire(server_state);
    defer generation_guard.release();
    const resources = manager.currentResources();
    const engine = &resources.engine;
    const tokenizer = &resources.tokenizer;
    const model_name = resources.display_name;

    const roles = allocator.alloc([]const u8, messages.len) catch {
        try conn.sendError(500, "internal_error", "Out of memory");
        return;
    };
    defer allocator.free(roles);
    const contents = allocator.alloc([]const u8, messages.len) catch {
        try conn.sendError(500, "internal_error", "Out of memory");
        return;
    };
    defer allocator.free(contents);

    var prompt_cap: usize = 32;
    for (messages, 0..) |msg, i| {
        roles[i] = normalizeRole(msg.role);
        contents[i] = msg.content;
        prompt_cap += roles[i].len + msg.content.len + 32;
    }

    const prompt_buf = allocator.alloc(u8, prompt_cap) catch {
        try conn.sendError(500, "internal_error", "Out of memory");
        return;
    };
    defer allocator.free(prompt_buf);

    const prompt = tokenizer.applyChatTemplate(roles, contents, prompt_buf) catch {
        try conn.sendError(400, "invalid_request_error", "Prompt too long");
        return;
    };

    // Tokenize
    // `encode` uses the tokenizer's allocator, which differs from the per-request
    // page allocator in server mode. Keep BOS packing in `encodePrompt` so this
    // route cannot accidentally free tokenizer-owned memory with the wrong allocator.
    const prompt_tokens = tokenizer.encodePrompt(prompt, allocator) catch {
        try conn.sendError(500, "internal_error", "Tokenization failed");
        return;
    };
    errdefer allocator.free(prompt_tokens);
    defer allocator.free(prompt_tokens);
    if (prompt_tokens.len == 0) {
        try conn.sendError(500, "internal_error", "Tokenization produced no prompt tokens");
        return;
    }
    const ts = @divTrunc(std.time.timestamp(), 1);
    const max_tokens = parsed.max_tokens;
    const req_id = "chatcmpl-zinc0001"; // TODO: T013 unique IDs

    if (parsed.stream) {
        // Streaming path — per-token SSE delivery
        conn.sendSseStart() catch return;

        // Send first chunk with role
        {
            var chunk_buf: [1024]u8 = undefined;
            const chunk = std.fmt.bufPrint(&chunk_buf,
                \\{{"id":"{s}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"role":"assistant"}},"finish_reason":null}}]}}
            , .{ req_id, ts, model_name }) catch return;
            conn.writeSseEvent(chunk) catch return;
        }
        if (conn.isPeerClosed()) return;

        // Prefill prompt tokens
        var state = forward_mod.DecodeState.init(allocator);
        defer state.deinit();
        engine.prefillBatch(&state, prompt_tokens) catch {
            conn.writeSseDone() catch {};
            return;
        };
        if (conn.isPeerClosed()) return;

        // Decode loop with buffered stop detection.
        // Tokens are buffered and only sent once we confirm they're not part of <|im_end|>.
        const eos = tokenizer.eosId();
        const stop_strs = chat_stop_strs[0..];
        var pending_tokens: [16]u32 = undefined; // tokens waiting to be sent
        var pending_count: usize = 0;
        var gen_text_buf: [4096]u8 = undefined; // accumulated decoded text for stop check
        var gen_text_len: usize = 0;
        var sent_text_len: usize = 0; // how much of gen_text has been confirmed safe to send
        var stopped = false;

        var generated: u32 = 0;
        if (max_tokens > 0) {
            var prev_token = engine.sampleGreedy();
            generated = 1;

            while (generated <= max_tokens and prev_token != eos and !stopped) {
                if (conn.isPeerClosed()) return;

                // Accumulate this token's decoded text
                var dec_buf: [256]u8 = undefined;
                const tok_text = tokenizer.decodeToken(prev_token, &dec_buf);
                if (isReplacementArtifact(tok_text)) {
                    if (generated < max_tokens) {
                        if (conn.isPeerClosed()) return;
                        engine.decodeStep(&state, prev_token, true) catch break;
                        if (conn.isPeerClosed()) return;
                        prev_token = engine.sampleGreedy();
                        generated += 1;
                        continue;
                    }
                    break;
                }
                if (gen_text_len + tok_text.len < gen_text_buf.len) {
                    @memcpy(gen_text_buf[gen_text_len..][0..tok_text.len], tok_text);
                    gen_text_len += tok_text.len;
                }

                // Add to pending queue
                if (pending_count < pending_tokens.len) {
                    pending_tokens[pending_count] = prev_token;
                    pending_count += 1;
                }

                // Check for full stop match against all stop strings
                if (findFirstStop(gen_text_buf[0..gen_text_len], stop_strs)) |_| {
                    stopped = true;
                }
                if (stopped) break;

                // Check if any suffix could be a prefix of any stop string
                var is_partial = false;
                for (stop_strs) |ss| {
                    const check_len = @min(gen_text_len, ss.len - 1);
                    var sl: usize = 1;
                    while (sl <= check_len) : (sl += 1) {
                        const suffix = gen_text_buf[gen_text_len - sl .. gen_text_len];
                        if (std.mem.startsWith(u8, ss, suffix)) {
                            is_partial = true;
                            break;
                        }
                    }
                    if (is_partial) break;
                }
                if (!is_partial and hasDanglingTrailingQuote(gen_text_buf[sent_text_len..gen_text_len])) {
                    is_partial = true;
                }

                if (!is_partial) {
                    // Safe to send all pending tokens
                    for (pending_tokens[0..pending_count]) |tid| {
                        streamToken(conn, tid, tokenizer, req_id, ts, model_name) catch return;
                    }
                    pending_count = 0;
                    sent_text_len = gen_text_len;
                }

                // Generate next token
                if (generated < max_tokens) {
                    if (conn.isPeerClosed()) return;
                    engine.decodeStep(&state, prev_token, true) catch break;
                    if (conn.isPeerClosed()) return;
                    prev_token = engine.sampleGreedy();
                    generated += 1;
                } else break;
            }

            // Flush any remaining pending tokens (only if we didn't hit stop)
            if (!stopped) {
                const pending_text = gen_text_buf[sent_text_len..gen_text_len];
                const cleaned_pending = trimTrailingChatArtifacts(pending_text);
                if (cleaned_pending.len > 0) {
                    streamText(conn, cleaned_pending, req_id, ts, model_name) catch return;
                }
            }
        }

        // Final chunk with finish_reason
        {
            var chunk_buf: [1024]u8 = undefined;
            const chunk = std.fmt.bufPrint(&chunk_buf,
                \\{{"id":"{s}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}]}}
            , .{ req_id, ts, model_name }) catch "";
            conn.writeSseEvent(chunk) catch return;
        }

        conn.writeSseDone() catch return;
    } else {
        // Non-streaming: use same prefill+decode loop with stop detection
        var state2 = forward_mod.DecodeState.init(allocator);
        defer state2.deinit();
        engine.prefillBatch(&state2, prompt_tokens) catch {
            try conn.sendError(500, "internal_error", "Prefill failed");
            return;
        };
        var text_buf: std.ArrayList(u8) = .{};
        defer text_buf.deinit(allocator);
        var ns_gen: u32 = 0;
        const ns_eos = tokenizer.eosId();
        const ns_stops = chat_stop_strs[0..];
        if (max_tokens > 0) {
            var prev = engine.sampleGreedy();
            ns_gen = 1;
            while (ns_gen <= max_tokens and prev != ns_eos) {
                var decode_buf2: [256]u8 = undefined;
                const tok_utf8 = tokenizer.decodeToken(prev, &decode_buf2);
                if (isReplacementArtifact(tok_utf8)) {
                    engine.decodeStep(&state2, prev, true) catch break;
                    prev = engine.sampleGreedy();
                    ns_gen += 1;
                    continue;
                }
                text_buf.appendSlice(allocator, tok_utf8) catch break;
                const hit = if (findFirstStop(text_buf.items, ns_stops)) |pos| blk: {
                    text_buf.shrinkRetainingCapacity(pos);
                    break :blk true;
                } else false;
                if (hit) break;
                engine.decodeStep(&state2, prev, true) catch break;
                prev = engine.sampleGreedy();
                ns_gen += 1;
            }
        }

        // Escape the full text for JSON
        var escaped_buf: [16384]u8 = undefined;
        const escaped_text = jsonEscape(trimTrailingChatArtifacts(text_buf.items), &escaped_buf);

        var resp_buf: [32768]u8 = undefined;
        const resp = std.fmt.bufPrint(&resp_buf,
            \\{{"id":"{s}","object":"chat.completion","created":{d},"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
        , .{
            req_id,                     ts,                model_name,
            escaped_text,               prompt_tokens.len, ns_gen,
            prompt_tokens.len + ns_gen,
        }) catch {
            try conn.sendError(500, "internal_error", "Response too large");
            return;
        };
        try conn.sendJson(200, resp);
    }
}

// ── /v1/completions ──────────────────────────────────────────

fn handleCompletions(
    conn: *http.Connection,
    manager: *model_manager_mod.ModelManager,
    server_state: *ServerState,
    body: []const u8,
    allocator: std.mem.Allocator,
) !void {
    var parsed = parseRequestBody(body, allocator) catch {
        try conn.sendError(400, "invalid_request_error", "Invalid JSON");
        return;
    };
    defer parsed.deinit();

    if (parsed.prompt.len == 0) {
        try conn.sendError(400, "invalid_request_error", "Field 'prompt' is required");
        return;
    }

    var generation_guard = GenerationGuard.acquire(server_state);
    defer generation_guard.release();
    const resources = manager.currentResources();
    const tokenizer = &resources.tokenizer;
    const engine = &resources.engine;
    const model_name = resources.display_name;

    // Tokenize raw prompt (no chat template)
    const prompt_tokens = tokenizer.encodePrompt(parsed.prompt, allocator) catch {
        try conn.sendError(500, "internal_error", "Tokenization failed");
        return;
    };
    defer allocator.free(prompt_tokens);
    if (prompt_tokens.len == 0) {
        try conn.sendError(500, "internal_error", "Tokenization produced no prompt tokens");
        return;
    }

    const ts = @divTrunc(std.time.timestamp(), 1);
    const req_id = "cmpl-zinc0001";

    const output_tokens = forward_mod.generate(engine, prompt_tokens, parsed.max_tokens, tokenizer.eosId(), allocator) catch {
        try conn.sendError(500, "internal_error", "Generation failed");
        return;
    };
    defer allocator.free(output_tokens);

    var text_buf: std.ArrayList(u8) = .{};
    defer text_buf.deinit(allocator);
    for (output_tokens) |tid| {
        const t = if (tid < tokenizer.vocab.len) tokenizer.vocab[tid] else "<?>";
        text_buf.appendSlice(allocator, t) catch break;
    }

    var escaped_buf: [16384]u8 = undefined;
    const escaped_text = jsonEscape(text_buf.items, &escaped_buf);

    var resp_buf: [32768]u8 = undefined;
    const resp = std.fmt.bufPrint(&resp_buf,
        \\{{"id":"{s}","object":"text_completion","created":{d},"model":"{s}","choices":[{{"index":0,"text":"{s}","finish_reason":"stop"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
    , .{
        req_id,                                ts,                model_name,
        escaped_text,                          prompt_tokens.len, output_tokens.len,
        prompt_tokens.len + output_tokens.len,
    }) catch {
        try conn.sendError(500, "internal_error", "Response too large");
        return;
    };
    try conn.sendJson(200, resp);
}

// ── Helpers ──────────────────────────────────────────────────

const RawMessage = struct {
    role: []const u8 = "",
    content: []const u8 = "",
};

const RawRequestBody = struct {
    model: []const u8 = "",
    messages: ?[]RawMessage = null,
    prompt: []const u8 = "",
    max_tokens: u32 = 256,
    stream: bool = false,
    temperature: f32 = 1.0,
};

const JsonMessage = struct {
    role: []u8 = &[_]u8{},
    content: []u8 = &[_]u8{},
};

const ParsedRequest = struct {
    model_id: []u8 = &[_]u8{},
    messages: []JsonMessage = &[_]JsonMessage{},
    prompt: []u8 = &[_]u8{},
    max_tokens: u32 = 256,
    stream: bool = false,
    temperature: f32 = 1.0,
    allocator: std.mem.Allocator,

    fn deinit(self: *ParsedRequest) void {
        if (self.model_id.len > 0) self.allocator.free(self.model_id);
        for (self.messages) |msg| {
            if (msg.role.len > 0) self.allocator.free(msg.role);
            if (msg.content.len > 0) self.allocator.free(msg.content);
        }
        if (self.messages.len > 0) self.allocator.free(self.messages);
        if (self.prompt.len > 0) self.allocator.free(self.prompt);
        self.* = undefined;
    }
};

fn parseRequestBody(body: []const u8, allocator: std.mem.Allocator) !ParsedRequest {
    const parsed = try std.json.parseFromSlice(RawRequestBody, allocator, body, .{
        .ignore_unknown_fields = true,
        .duplicate_field_behavior = .use_last,
        .allocate = .alloc_always,
    });
    defer parsed.deinit();

    var result = ParsedRequest{
        .allocator = allocator,
    };
    errdefer result.deinit();

    result.model_id = try decodeJsonText(allocator, parsed.value.model);
    result.max_tokens = parsed.value.max_tokens;
    result.stream = parsed.value.stream;
    result.temperature = parsed.value.temperature;
    result.prompt = try decodeJsonText(allocator, parsed.value.prompt);

    if (parsed.value.messages) |messages| {
        result.messages = try allocator.alloc(JsonMessage, messages.len);
        for (messages, 0..) |msg, i| {
            result.messages[i] = .{
                .role = try decodeJsonText(allocator, msg.role),
                .content = try decodeJsonText(allocator, msg.content),
            };
        }
    }

    return result;
}

fn decodeJsonText(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    if (input.len == 0) return &[_]u8{};

    var out: std.ArrayList(u8) = .{};
    defer out.deinit(allocator);

    var i: usize = 0;
    while (i < input.len) : (i += 1) {
        if (input[i] != '\\' or i + 1 >= input.len) {
            try out.append(allocator, input[i]);
            continue;
        }

        i += 1;
        switch (input[i]) {
            '"' => try out.append(allocator, '"'),
            '\\' => try out.append(allocator, '\\'),
            '/' => try out.append(allocator, '/'),
            'b' => try out.append(allocator, 0x08),
            'f' => try out.append(allocator, 0x0c),
            'n' => try out.append(allocator, '\n'),
            'r' => try out.append(allocator, '\r'),
            't' => try out.append(allocator, '\t'),
            else => try out.append(allocator, input[i]),
        }
    }

    return out.toOwnedSlice(allocator);
}

fn normalizeRole(role: []const u8) []const u8 {
    if (std.mem.eql(u8, role, "system") or
        std.mem.eql(u8, role, "user") or
        std.mem.eql(u8, role, "assistant"))
    {
        return role;
    }
    return "user";
}

test "decodeJsonText handles common escape sequences" {
    const decoded = try decodeJsonText(std.testing.allocator, "hello\\n\\\"world\\\"\\\\");
    defer if (decoded.len > 0) std.testing.allocator.free(decoded);
    try std.testing.expectEqualStrings("hello\n\"world\"\\", decoded);
}

test "decodeJsonText empty string" {
    const decoded = try decodeJsonText(std.testing.allocator, "");
    defer if (decoded.len > 0) std.testing.allocator.free(decoded);
    try std.testing.expectEqual(@as(usize, 0), decoded.len);
}

test "parseRequestBody extracts stream flag and user message" {
    const body = "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}";
    var parsed = try parseRequestBody(body, std.testing.allocator);
    defer parsed.deinit();
    try std.testing.expect(parsed.stream);
    try std.testing.expectEqualStrings("qwen", parsed.model_id);
    try std.testing.expectEqual(@as(usize, 1), parsed.messages.len);
    try std.testing.expectEqualStrings("user", parsed.messages[0].role);
    try std.testing.expectEqualStrings("hello", parsed.messages[0].content);
}

test "parseRequestBody extracts max_tokens and prompt" {
    const body = "{\"model\":\"qwen\",\"prompt\":\"test\",\"max_tokens\":128}";
    var parsed = try parseRequestBody(body, std.testing.allocator);
    defer parsed.deinit();
    try std.testing.expectEqualStrings("qwen", parsed.model_id);
    try std.testing.expectEqual(@as(u32, 128), parsed.max_tokens);
    try std.testing.expectEqualStrings("test", parsed.prompt);
}

test "parseRequestBody defaults when fields missing" {
    const body = "{\"model\":\"qwen\"}";
    var parsed = try parseRequestBody(body, std.testing.allocator);
    defer parsed.deinit();
    try std.testing.expectEqualStrings("qwen", parsed.model_id);
    try std.testing.expect(!parsed.stream);
    try std.testing.expectEqual(@as(u32, 256), parsed.max_tokens);
    try std.testing.expectEqual(@as(usize, 0), parsed.messages.len);
    try std.testing.expectEqualStrings("", parsed.prompt);
}

test "parseRequestBody handles escaped content and multiple messages" {
    const body =
        \\{"messages":[{"role":"assistant","content":"literal \"content\":\"noise\""},{"role":"user","content":"line1\\nline2"}],"stream":true}
    ;
    var parsed = try parseRequestBody(body, std.testing.allocator);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 2), parsed.messages.len);
    try std.testing.expectEqualStrings("assistant", parsed.messages[0].role);
    try std.testing.expectEqualStrings("literal \"content\":\"noise\"", parsed.messages[0].content);
    try std.testing.expectEqualStrings("line1\nline2", parsed.messages[1].content);
}

test "normalizeRole falls back to user" {
    try std.testing.expectEqualStrings("user", normalizeRole("tool"));
    try std.testing.expectEqualStrings("assistant", normalizeRole("assistant"));
}

test "ParsedRequest defaults" {
    var result = ParsedRequest{
        .allocator = std.testing.allocator,
    };
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 0), result.model_id.len);
    try std.testing.expectEqual(@as(usize, 0), result.messages.len);
    try std.testing.expectEqual(@as(usize, 0), result.prompt.len);
    try std.testing.expectEqual(@as(u32, 256), result.max_tokens);
    try std.testing.expect(!result.stream);
    try std.testing.expectEqual(@as(f32, 1.0), result.temperature);
}

fn jsonEscape(input: []const u8, buf: []u8) []const u8 {
    var out: usize = 0;
    for (input) |c| {
        if (out + 2 >= buf.len) break;
        switch (c) {
            '"' => {
                buf[out] = '\\';
                buf[out + 1] = '"';
                out += 2;
            },
            '\\' => {
                buf[out] = '\\';
                buf[out + 1] = '\\';
                out += 2;
            },
            '\n' => {
                buf[out] = '\\';
                buf[out + 1] = 'n';
                out += 2;
            },
            '\r' => {
                buf[out] = '\\';
                buf[out + 1] = 'r';
                out += 2;
            },
            '\t' => {
                buf[out] = '\\';
                buf[out + 1] = 't';
                out += 2;
            },
            else => {
                buf[out] = c;
                out += 1;
            },
        }
    }
    return buf[0..out];
}

/// Send a single token as an SSE ChatCompletionChunk event.
fn streamToken(
    conn: *http.Connection,
    token_id: u32,
    tokenizer: *const tokenizer_mod.Tokenizer,
    req_id: []const u8,
    ts: i64,
    model_name: []const u8,
) !void {
    // Decode GPT-2 byte encoding to real UTF-8
    var decode_buf: [256]u8 = undefined;
    const token_text = tokenizer.decodeToken(token_id, &decode_buf);
    try streamText(conn, token_text, req_id, ts, model_name);
}

fn streamText(
    conn: *http.Connection,
    text: []const u8,
    req_id: []const u8,
    ts: i64,
    model_name: []const u8,
) !void {
    var escaped_buf: [512]u8 = undefined;
    const escaped = jsonEscape(text, &escaped_buf);
    var chunk_buf: [1024]u8 = undefined;
    const chunk = std.fmt.bufPrint(&chunk_buf,
        \\{{"id":"{s}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"content":"{s}"}},"finish_reason":null}}]}}
    , .{ req_id, ts, model_name, escaped }) catch return error.BufferTooSmall;
    try conn.writeSseEvent(chunk);
}

// ── Built-in Chat UI ─────────────────────────────────────────

fn serveChatUi(conn: *http.Connection) !void {
    const html = @embedFile("chat.html");
    var buf: [256]u8 = undefined;
    const header = std.fmt.bufPrint(&buf, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{html.len}) catch return error.HeaderTooLarge;
    try conn.stream.writeAll(header);
    try conn.stream.writeAll(html);
}

fn fallbackModelName(model: *const Model) []const u8 {
    return switch (model.config.architecture) {
        .qwen35 => "qwen3.5",
        .qwen2_moe => "qwen3.5-35b",
        .qwen2 => "qwen2",
        .llama => "llama",
        .mistral => "mistral",
        .mamba => "mamba",
        .jamba => "jamba",
        .unknown => "zinc-model",
    };
}

fn modelName(model: *const Model) []const u8 {
    return model.gguf_file.getString("general.basename") orelse
        model.gguf_file.getString("general.name") orelse
        fallbackModelName(model);
}

test "jsonEscape handles special characters" {
    var buf: [64]u8 = undefined;
    const result = jsonEscape("hello \"world\"\nfoo", &buf);
    try std.testing.expectEqualStrings("hello \\\"world\\\"\\nfoo", result);
}

test "jsonEscape handles tabs and backslashes" {
    var buf: [64]u8 = undefined;
    const result = jsonEscape("a\\b\tc", &buf);
    try std.testing.expectEqualStrings("a\\\\b\\tc", result);
}

test "jsonEscape empty string" {
    var buf: [64]u8 = undefined;
    const result = jsonEscape("", &buf);
    try std.testing.expectEqualStrings("", result);
}

test "jsonEscape carriage return" {
    var buf: [32]u8 = undefined;
    const result = jsonEscape("a\rb", &buf);
    try std.testing.expectEqualStrings("a\\rb", result);
}

test "jsonEscape plain ASCII passthrough" {
    var buf: [64]u8 = undefined;
    const result = jsonEscape("Hello, World! 123", &buf);
    try std.testing.expectEqualStrings("Hello, World! 123", result);
}

test "trimTrailingChatArtifacts strips endoftext and replacement junk" {
    const text = "Hello!\n \xEF\xBF\xBD\xEF\xBF\xBD<|endoftext|>\n\n";
    try std.testing.expectEqualStrings("Hello!", trimTrailingChatArtifacts(text));
}

test "trimTrailingChatArtifacts strips unmatched trailing quote after punctuation" {
    try std.testing.expectEqualStrings("Hello! How can I help you today?", trimTrailingChatArtifacts("Hello! How can I help you today?\"\n\n"));
    try std.testing.expectEqualStrings("\"Paris.\"", trimTrailingChatArtifacts("\"Paris.\""));
}

test "isReplacementArtifact detects replacement-only chunks" {
    try std.testing.expect(isReplacementArtifact(" \xEF\xBF\xBD"));
    try std.testing.expect(isReplacementArtifact("\xEF\xBF\xBD\xEF\xBF\xBD"));
    try std.testing.expect(!isReplacementArtifact("Hello \xEF\xBF\xBD"));
}

test "hasDanglingTrailingQuote detects unmatched punctuation quote suffix" {
    try std.testing.expect(hasDanglingTrailingQuote("Hello?\""));
    try std.testing.expect(hasDanglingTrailingQuote("Hello?\"\n\n"));
    try std.testing.expect(!hasDanglingTrailingQuote("\"Paris.\""));
    try std.testing.expect(!hasDanglingTrailingQuote("He said \"hi\""));
}

test "ServerState snapshot tracks active queued and uptime" {
    var state = ServerState.init(100);
    _ = state.active_requests.fetchAdd(1, .monotonic);
    _ = state.queued_requests.fetchAdd(2, .monotonic);

    const snapshot = state.snapshot(112);
    try std.testing.expectEqual(@as(u32, 1), snapshot.active_requests);
    try std.testing.expectEqual(@as(u32, 2), snapshot.queued_requests);
    try std.testing.expectEqual(@as(u64, 12), snapshot.uptime_seconds);
}

test "buildHealthJson includes request counts and uptime" {
    var state = ServerState.init(std.time.timestamp() - 5);
    _ = state.active_requests.fetchAdd(1, .monotonic);
    _ = state.queued_requests.fetchAdd(1, .monotonic);

    var buf: [256]u8 = undefined;
    const body = try buildHealthJson(&state, "qwen3.5-35b", .{
        .device_local_bytes = 21 * 1024 * 1024 * 1024,
        .device_local_budget_bytes = 32 * 1024 * 1024 * 1024,
    }, &buf);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"status\":\"ok\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"model\":\"qwen3.5-35b\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"active_requests\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"queued_requests\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"uptime_seconds\":") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"gpu_memory_used_bytes\":") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"gpu_memory_budget_bytes\":") != null);
}
test "findFirstStop picks earliest chat control marker" {
    const text = "Hello<|im_start|>assistant<|im_end|>";
    try std.testing.expectEqual(@as(?usize, 5), findFirstStop(text, chat_stop_strs[0..]));
}

test "findFirstStop returns null when no chat stop marker exists" {
    try std.testing.expectEqual(@as(?usize, null), findFirstStop("Hello there", chat_stop_strs[0..]));
}

test "findFirstStop detects endoftext marker" {
    const text = "Hello<|endoftext|>";
    try std.testing.expectEqual(@as(?usize, 5), findFirstStop(text, chat_stop_strs[0..]));
}
