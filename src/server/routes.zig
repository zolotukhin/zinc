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

const chat_reuse_max_sessions: usize = 32;
const chat_reuse_idle_timeout_ns: i128 = 30 * 60 * std.time.ns_per_s;

const ChatReuseEntry = struct {
    session_id: []u8,
    model_path: []u8,
    prompt_tokens: []u32,
    last_used_ns: i128,

    fn deinit(self: *ChatReuseEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.session_id);
        allocator.free(self.model_path);
        allocator.free(self.prompt_tokens);
        self.* = undefined;
    }
};

const ChatReuseCache = struct {
    allocator: std.mem.Allocator,
    entries: std.ArrayListUnmanaged(ChatReuseEntry) = .{},

    fn init(allocator: std.mem.Allocator) ChatReuseCache {
        return .{ .allocator = allocator };
    }

    fn clear(self: *ChatReuseCache) void {
        for (self.entries.items) |*entry| entry.deinit(self.allocator);
        self.entries.clearAndFree(self.allocator);
    }

    fn deinit(self: *ChatReuseCache) void {
        self.clear();
    }

    fn removeAt(self: *ChatReuseCache, idx: usize) void {
        var entry = self.entries.swapRemove(idx);
        entry.deinit(self.allocator);
    }

    fn findSessionIndex(self: *const ChatReuseCache, session_id: []const u8) ?usize {
        for (self.entries.items, 0..) |entry, idx| {
            if (std.mem.eql(u8, entry.session_id, session_id)) return idx;
        }
        return null;
    }

    fn pruneExpired(self: *ChatReuseCache, now_ns: i128) void {
        var i: usize = 0;
        while (i < self.entries.items.len) {
            if (now_ns - self.entries.items[i].last_used_ns >= chat_reuse_idle_timeout_ns) {
                self.removeAt(i);
                continue;
            }
            i += 1;
        }
    }

    fn evictLru(self: *ChatReuseCache) void {
        if (self.entries.items.len == 0) return;
        var oldest_idx: usize = 0;
        var oldest_ns = self.entries.items[0].last_used_ns;
        for (self.entries.items[1..], 1..) |entry, idx| {
            if (entry.last_used_ns < oldest_ns) {
                oldest_ns = entry.last_used_ns;
                oldest_idx = idx;
            }
        }
        self.removeAt(oldest_idx);
    }

    fn matchingPrefixLen(self: *ChatReuseCache, session_id: []const u8, model_path: []const u8, prompt_tokens: []const u32, now_ns: i128) usize {
        self.pruneExpired(now_ns);
        for (self.entries.items) |*entry| {
            if (!std.mem.eql(u8, entry.session_id, session_id)) continue;
            if (!std.mem.eql(u8, entry.model_path, model_path)) return 0;
            if (!std.mem.startsWith(u32, prompt_tokens, entry.prompt_tokens)) return 0;
            entry.last_used_ns = now_ns;
            return entry.prompt_tokens.len;
        }
        return 0;
    }

    fn removeSession(self: *ChatReuseCache, session_id: []const u8) void {
        if (self.findSessionIndex(session_id)) |idx| {
            self.removeAt(idx);
        }
    }

    fn count(self: *const ChatReuseCache) usize {
        return self.entries.items.len;
    }

    fn store(self: *ChatReuseCache, session_id: []const u8, model_path: []const u8, prompt_tokens: []const u32, now_ns: i128) !void {
        self.pruneExpired(now_ns);
        self.removeSession(session_id);
        while (self.entries.items.len >= chat_reuse_max_sessions) {
            self.evictLru();
        }

        const entry = ChatReuseEntry{
            .session_id = try self.allocator.dupe(u8, session_id),
            .model_path = try self.allocator.dupe(u8, model_path),
            .prompt_tokens = try self.allocator.dupe(u32, prompt_tokens),
            .last_used_ns = now_ns,
        };
        errdefer {
            var owned = entry;
            owned.deinit(self.allocator);
        }
        try self.entries.append(self.allocator, entry);
    }
};

pub const ServerState = struct {
    started_at: i64,
    active_requests: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    queued_requests: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    active_context_tokens: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    generation_mutex: std.Thread.Mutex = .{},
    downloads: DownloadTracker = .{},
    chat_reuse_cache: ChatReuseCache,

    pub fn init(started_at: i64) ServerState {
        return .{
            .started_at = started_at,
            .chat_reuse_cache = ChatReuseCache.init(std.heap.page_allocator),
        };
    }

    pub fn deinit(self: *ServerState) void {
        self.chat_reuse_cache.deinit();
    }

    pub fn uptimeSeconds(self: *const ServerState, now: i64) u64 {
        return @intCast(@max(now - self.started_at, 0));
    }

    pub fn snapshot(self: *const ServerState, now: i64) HealthSnapshot {
        return .{
            .active_requests = self.active_requests.load(.monotonic),
            .queued_requests = self.queued_requests.load(.monotonic),
            .active_context_tokens = self.active_context_tokens.load(.monotonic),
            .uptime_seconds = self.uptimeSeconds(now),
        };
    }

    pub fn setActiveContextTokens(self: *ServerState, tokens: u32) void {
        self.active_context_tokens.store(tokens, .monotonic);
    }

    pub fn clearActiveContext(self: *ServerState) void {
        self.active_context_tokens.store(0, .monotonic);
    }

    pub fn clearChatReuseCache(self: *ServerState) void {
        self.chat_reuse_cache.clear();
    }

    pub fn clearChatReuseSession(self: *ServerState, session_id: []const u8) void {
        self.chat_reuse_cache.removeSession(session_id);
    }
};

const HealthSnapshot = struct {
    active_requests: u32,
    queued_requests: u32,
    active_context_tokens: u32,
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
        try handleActivateModel(conn, manager, server_state, request.body);
    } else if (request.method == .POST and std.mem.eql(u8, request.path, "/v1/models/pull")) {
        try handlePullModel(conn, manager, server_state, request.body, allocator);
    } else if (request.method == .POST and std.mem.eql(u8, request.path, "/v1/models/remove")) {
        try handleRemoveModel(conn, manager, server_state, request.body);
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
    const active_context_tokens = memory_usage.activeContextTokens(snapshot.active_context_tokens);
    const active_context_bytes = memory_usage.activeContextBytes(active_context_tokens);
    return std.fmt.bufPrint(buf,
        \\{{"status":"ok","model":"{s}","active_requests":{d},"queued_requests":{d},"uptime_seconds":{d},"gpu_memory_used_bytes":{d},"gpu_memory_budget_bytes":{d},"gpu_memory_weights_bytes":{d},"gpu_memory_runtime_bytes":{d},"gpu_context_reserved_bytes":{d},"gpu_context_active_bytes":{d},"gpu_context_tokens":{d},"gpu_context_capacity_tokens":{d}}}
    , .{
        model_name,
        snapshot.active_requests,
        snapshot.queued_requests,
        snapshot.uptime_seconds,
        memory_usage.device_local_bytes,
        memory_usage.device_local_budget_bytes,
        memory_usage.weights_bytes,
        memory_usage.runtime_device_local_bytes,
        memory_usage.context_reserved_bytes,
        active_context_bytes,
        active_context_tokens,
        memory_usage.context_capacity_tokens,
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
        "{{\"object\":\"list\",\"profile\":\"{s}\",\"active_memory_used_bytes\":{d},\"active_memory_budget_bytes\":{d},\"active_memory_weights_bytes\":{d},\"active_memory_runtime_bytes\":{d},\"active_context_reserved_bytes\":{d},\"active_context_active_bytes\":{d},\"active_context_tokens\":{d},\"active_context_capacity_tokens\":{d},\"data\":[",
        .{
            view.profile,
            memory_usage.device_local_bytes,
            memory_usage.device_local_budget_bytes,
            memory_usage.weights_bytes,
            memory_usage.runtime_device_local_bytes,
            memory_usage.context_reserved_bytes,
            memory_usage.activeContextBytes(server_state.active_context_tokens.load(.monotonic)),
            memory_usage.activeContextTokens(server_state.active_context_tokens.load(.monotonic)),
            memory_usage.context_capacity_tokens,
        },
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
            \\{{"id":"{s}","object":"model","created":{d},"owned_by":"zinc","display_name":"{s}","release_date":"{s}","homepage_url":"{s}","installed":{s},"active":{s},"managed":{s},"supported_on_current_gpu":{s},"fits_current_gpu":{s},"required_vram_bytes":{d},"fit_source":"{s}","status":"{s}","supports_thinking_toggle":{s},"downloading":{s},"download_phase":"{s}","downloaded_bytes":{d},"download_total_bytes":{d},"download_error":"{s}"}} 
        , .{
            entry.id,
            ts,
            entry.display_name,
            entry.release_date,
            entry.homepage_url,
            if (entry.installed) "true" else "false",
            if (entry.active) "true" else "false",
            if (entry.managed) "true" else "false",
            if (entry.supported_on_current_gpu) "true" else "false",
            if (entry.fits_current_gpu) "true" else "false",
            entry.required_vram_bytes,
            fit_source,
            entry.status_label,
            if (entry.supports_thinking_toggle) "true" else "false",
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
) !void {
    const parsed = parseJsonFields(body) catch {
        try conn.sendError(400, "invalid_request_error", "Invalid JSON in request body");
        return;
    };
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
    server_state.clearChatReuseCache();

    var buf: [512]u8 = undefined;
    const response = std.fmt.bufPrint(&buf,
        \\{{"object":"model.activation","id":"{s}","active":true}}
    , .{parsed.model_id}) catch return error.BufferTooSmall;
    try conn.sendJson(200, response);
}

fn handleRemoveModel(
    conn: *http.Connection,
    manager: *model_manager_mod.ModelManager,
    server_state: *ServerState,
    body: []const u8,
) !void {
    const parsed = parseJsonFields(body) catch {
        try conn.sendError(400, "invalid_request_error", "Invalid JSON in request body");
        return;
    };
    if (parsed.model_id.len == 0) {
        try conn.sendError(400, "invalid_request_error", "Field 'model' is required");
        return;
    }
    if (catalog_mod.find(parsed.model_id) == null) {
        try conn.sendError(400, "invalid_request_error", "Unknown managed model id");
        return;
    }

    server_state.generation_mutex.lock();
    defer server_state.generation_mutex.unlock();

    const result = manager.removeManagedModel(parsed.model_id, parsed.force) catch |err| {
        const msg = switch (err) {
            error.ModelNotInstalled => "Model is not installed in the local cache",
            error.ModelLoadedInGpu => "Model is currently loaded in GPU memory. Retry with force=true to unload it first.",
            else => "Model removal failed",
        };
        const status: u16 = switch (err) {
            error.ModelLoadedInGpu => 409,
            else => 400,
        };
        try conn.sendError(status, "invalid_request_error", msg);
        return;
    };
    server_state.clearChatReuseCache();

    var buf: [768]u8 = undefined;
    const response = std.fmt.bufPrint(&buf,
        \\{{"object":"model.remove","id":"{s}","removed":true,"unloaded_from_gpu":{s},"cleared_active_selection":{s},"deleted_model":{s},"deleted_manifest":{s},"removed_dir":{s}}}
    , .{
        parsed.model_id,
        if (result.unloaded_from_gpu) "true" else "false",
        if (result.cleared_active_selection) "true" else "false",
        if (result.removed.deleted_model) "true" else "false",
        if (result.removed.deleted_manifest) "true" else "false",
        if (result.removed.removed_dir) "true" else "false",
    }) catch return error.BufferTooSmall;
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
    const parsed = parseJsonFields(body) catch {
        try conn.sendError(400, "invalid_request_error", "Invalid JSON in request body");
        return;
    };
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
const leaked_reasoning_markers = [_][]const u8{
    "\n\nThe user is asking",
    "\n\nThe user's request",
    "\n\nHere is the response:",
    "\n\nI will provide a direct answer",
    "\n\nI will provide the response",
    "\n\nI will answer directly",
    "\nSince I am acting as the assistant",
    " Since I am acting as the assistant",
    "\nHowever, looking at the prompt structure",
    " However, looking at the prompt structure",
    "\nWait, looking closely at the prompt structure",
    " Wait, looking closely at the prompt structure",
};

const utf8_replacement = "\xEF\xBF\xBD";
const thinking_prefix = "<think>\n";
const empty_thinking_prefix = "<think>\n\n</think>\n\n";
const chat_history_answer_limit_bytes: usize = 640;
const default_chat_system_prompt =
    "Answer directly. If a user term is ambiguous or looks misspelled, say that briefly and continue with the most likely interpretation. Never output self-referential planning or phrases like 'I need to complete the response'.";

const FinishReason = enum {
    stop,
    length,
};

const ChatMessage = struct {
    role: []const u8 = "",
    content: []const u8 = "",
};

const ChatRequestBody = struct {
    model: []const u8 = "",
    session_id: []const u8 = "",
    messages: []const ChatMessage = &.{},
    max_tokens: u32 = 256,
    stream: bool = false,
    temperature: f32 = 0.0,
    top_p: f32 = 1.0,
    enable_thinking: ?bool = null,
};

const ParsedChatRequest = struct {
    parsed: std.json.Parsed(ChatRequestBody),
    roles: []const []const u8,
    contents: []const []const u8,
    session_id: []const u8,
    max_tokens: u32,
    stream: bool,
    temperature: f32,
    top_p: f32,
    enable_thinking: ?bool,

    fn deinit(self: *ParsedChatRequest) void {
        self.parsed.deinit();
        self.* = undefined;
    }
};

fn countValidChatMessages(messages: []const ChatMessage) usize {
    var count: usize = 0;
    for (messages) |message| {
        if (message.role.len == 0 or message.content.len == 0) continue;
        count += 1;
    }
    return count;
}

fn estimateChatPromptBytes(roles: []const []const u8, contents: []const []const u8, thinking_enabled: bool) usize {
    var total: usize = 128;
    const n = @min(roles.len, contents.len);
    for (0..n) |i| {
        total += roles[i].len + contents[i].len + 32;
    }
    if (thinking_enabled) total += thinking_prefix.len + 32 else total += 32;
    return total;
}

fn buildChatPrompt(tokenizer: *const tokenizer_mod.Tokenizer, roles: []const []const u8, contents: []const []const u8, enable_thinking: ?bool, skip_thinking_template: bool, buf: []u8) ![]const u8 {
    return tokenizer.applyChatTemplateWithOptions(roles, contents, .{ .enable_thinking = enable_thinking, .skip_thinking_template = skip_thinking_template }, buf);
}

fn buildChatTranscriptPrompt(tokenizer: *const tokenizer_mod.Tokenizer, roles: []const []const u8, contents: []const []const u8, enable_thinking: ?bool, buf: []u8) ![]const u8 {
    return tokenizer.applyChatTemplateWithOptions(roles, contents, .{
        .add_generation_prompt = false,
        .enable_thinking = enable_thinking,
    }, buf);
}

fn warmChatReuseCache(
    server_state: *ServerState,
    resources: *const model_manager_mod.LoadedResources,
    tokenizer: *const tokenizer_mod.Tokenizer,
    engine: *forward_mod.InferenceEngine,
    thinking_enabled: bool,
    session_id: []const u8,
    roles: []const []const u8,
    contents: []const []const u8,
    assistant_content: []const u8,
    state: *forward_mod.DecodeState,
    prompt_tokens: []const u32,
    processed_generated_tokens: []const u32,
    allocator: std.mem.Allocator,
) !void {
    if (session_id.len == 0) return;
    const now_ns = std.time.nanoTimestamp();

    const processed_prefix_len = prompt_tokens.len + processed_generated_tokens.len;
    const transcript_count = roles.len + 1;
    const transcript_roles = try allocator.alloc([]const u8, transcript_count);
    defer allocator.free(transcript_roles);
    const transcript_contents = try allocator.alloc([]const u8, transcript_count);
    defer allocator.free(transcript_contents);

    @memcpy(transcript_roles[0..roles.len], roles);
    @memcpy(transcript_contents[0..contents.len], contents);
    transcript_roles[roles.len] = "assistant";
    transcript_contents[contents.len] = assistant_content;

    const transcript_capacity = estimateChatPromptBytes(transcript_roles, transcript_contents, false) + assistant_content.len + 64;
    const transcript_buf = try allocator.alloc(u8, transcript_capacity);
    defer allocator.free(transcript_buf);
    const transcript_prompt = try buildChatTranscriptPrompt(tokenizer, transcript_roles, transcript_contents, thinking_enabled, transcript_buf);
    const transcript_tokens = try tokenizer.encodePrompt(transcript_prompt, allocator);
    defer allocator.free(transcript_tokens);

    const prompt_mismatch = if (transcript_tokens.len < prompt_tokens.len)
        prompt_tokens.len
    else
        firstTokenMismatch(transcript_tokens[0..prompt_tokens.len], prompt_tokens);
    const response_mismatch = if (transcript_tokens.len < processed_prefix_len or prompt_mismatch != null)
        @as(?usize, 0)
    else
        firstTokenMismatch(transcript_tokens[prompt_tokens.len..processed_prefix_len], processed_generated_tokens);

    const can_incremental =
        state.position == processed_prefix_len and
        transcript_tokens.len >= processed_prefix_len and
        prompt_mismatch == null and
        response_mismatch == null;

    if (can_incremental) {
        const suffix_tokens = transcript_tokens[processed_prefix_len..];
        if (suffix_tokens.len > 0) {
            try engine.prefillBatch(state, suffix_tokens);
        }
        log.info("chat cache updated: session={s} prefix={d} suffix={d}", .{
            session_id,
            transcript_tokens.len,
            transcript_tokens.len - processed_prefix_len,
        });
    } else {
        if (state.position != processed_prefix_len) {
            log.info("chat cache skipped: state position mismatch state={d} processed={d}", .{
                state.position,
                processed_prefix_len,
            });
        } else if (prompt_mismatch) |mismatch| {
            const transcript_token = if (mismatch < transcript_tokens.len) transcript_tokens[mismatch] else @as(u32, 0);
            const prompt_token = if (mismatch < prompt_tokens.len) prompt_tokens[mismatch] else @as(u32, 0);
            log.info("chat cache skipped after prompt mismatch: idx={d} transcript={d} prompt={d}", .{
                mismatch,
                transcript_token,
                prompt_token,
            });
        } else if (response_mismatch) |mismatch| {
            const transcript_slice = transcript_tokens[prompt_tokens.len..processed_prefix_len];
            const transcript_token = if (mismatch < transcript_slice.len) transcript_slice[mismatch] else @as(u32, 0);
            const processed_token = if (mismatch < processed_generated_tokens.len) processed_generated_tokens[mismatch] else @as(u32, 0);
            log.info("chat cache skipped after response mismatch: idx={d} transcript={d} processed={d} processed_len={d}", .{
                mismatch,
                transcript_token,
                processed_token,
                processed_generated_tokens.len,
            });
        }
        return error.CachePrefixMismatch;
    }

    try server_state.chat_reuse_cache.store(session_id, resources.model_path, transcript_tokens, now_ns);
}

fn firstTokenMismatch(a: []const u32, b: []const u32) ?usize {
    const n = @min(a.len, b.len);
    for (0..n) |i| {
        if (a[i] != b[i]) return i;
    }
    if (a.len != b.len) return n;
    return null;
}

fn parseChatRequest(allocator: std.mem.Allocator, body: []const u8) !ParsedChatRequest {
    var parsed = try std.json.parseFromSlice(ChatRequestBody, allocator, body, .{
        .ignore_unknown_fields = true,
    });
    errdefer parsed.deinit();

    const arena_allocator = parsed.arena.allocator();
    const messages = parsed.value.messages;
    const roles = try arena_allocator.alloc([]const u8, messages.len + 1);
    const contents = try arena_allocator.alloc([]const u8, messages.len + 1);

    var count: usize = 0;
    var has_guiding_message = false;
    for (messages) |message| {
        if (message.role.len == 0 or message.content.len == 0) continue;
        if (std.mem.eql(u8, message.role, "system") or std.mem.eql(u8, message.role, "developer")) {
            has_guiding_message = true;
        }
    }

    if (!has_guiding_message) {
        roles[count] = "system";
        contents[count] = default_chat_system_prompt;
        count += 1;
    }

    for (messages) |message| {
        if (message.role.len == 0 or message.content.len == 0) continue;
        roles[count] = message.role;
        contents[count] = if (std.mem.eql(u8, message.role, "assistant"))
            sanitizeAssistantHistoryContent(message.content)
        else
            message.content;
        count += 1;
    }

    return .{
        .parsed = parsed,
        .roles = roles[0..count],
        .contents = contents[0..count],
        .session_id = parsed.value.session_id,
        .max_tokens = parsed.value.max_tokens,
        .stream = parsed.value.stream,
        .temperature = parsed.value.temperature,
        .top_p = parsed.value.top_p,
        .enable_thinking = parsed.value.enable_thinking,
    };
}

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
        const trimmed_left = trimLeadingStandaloneQuote(out);
        const trimmed = std.mem.trimRight(u8, trimmed_left, " \t\r\n");
        if (std.mem.endsWith(u8, trimmed, "<|endoftext|>")) {
            out = trimmed[0 .. trimmed.len - "<|endoftext|>".len];
            continue;
        }
        if (std.mem.endsWith(u8, trimmed, utf8_replacement)) {
            out = trimmed[0 .. trimmed.len - utf8_replacement.len];
            continue;
        }
        if (hasDanglingTrailingQuote(trimmed)) {
            out = trimmed[0 .. trimmed.len - 1];
            continue;
        }
        if (trimDanglingHeading(trimmed)) |next| {
            out = next;
            continue;
        }
        if (trimDanglingListMarker(trimmed)) |next| {
            out = next;
            continue;
        }
        return trimmed;
    }
}

fn trimDanglingListMarker(text: []const u8) ?[]const u8 {
    if (text.len == 0) return null;
    const line_start = (std.mem.lastIndexOfScalar(u8, text, '\n') orelse return null) + 1;
    const line = std.mem.trim(u8, text[line_start..], " \t\r\n");
    if (line.len == 1 and (line[0] == '-' or line[0] == '*')) {
        return std.mem.trimRight(u8, text[0..line_start], " \t\r\n");
    }
    return null;
}

const leaked_reasoning_start_markers = [_][]const u8{
    "The user is asking",
    "The user's request",
    "The user's question",
    "I need to provide",
    "I need to answer",
    "I need to respond",
    "I should answer",
    "I should provide",
    "I should respond",
    "I will provide",
    "I will answer",
    "I will respond",
    "Let me answer",
    "Let me provide",
    "Let me respond",
    "Let me think",
    "Let me analyze",
};

fn startsWithLeakedReasoning(text: []const u8) bool {
    const trimmed = std.mem.trimLeft(u8, text, " \t\r\n");
    for (leaked_reasoning_start_markers) |marker| {
        if (trimmed.len >= marker.len and std.ascii.eqlIgnoreCase(trimmed[0..marker.len], marker)) return true;
    }
    return false;
}

fn findLeakedReasoningStart(text: []const u8) ?usize {
    var first: ?usize = null;
    for (leaked_reasoning_markers) |marker| {
        if (std.mem.indexOf(u8, text, marker)) |idx| {
            if (idx >= 48 and (first == null or idx < first.?)) first = idx;
        }
    }
    return first;
}

fn trimLeakedNoThinkingOutput(text: []const u8) []const u8 {
    if (findLeakedReasoningStart(text)) |idx| {
        return std.mem.trimRight(u8, text[0..idx], " \t\r\n");
    }
    return text;
}

fn findUnexpectedThinkingTailStart(text: []const u8) ?usize {
    // If text starts with <think>, skip past the first </think> so we detect REOPENED blocks
    var search_start: usize = 0;
    const trimmed_start = std.mem.trimLeft(u8, text, " \t\r\n");
    if (std.mem.startsWith(u8, trimmed_start, "<think>")) {
        if (std.mem.indexOf(u8, text, "</think>")) |close_idx| {
            search_start = close_idx + "</think>".len;
        } else {
            return null; // thinking block not closed yet
        }
    }
    if (std.mem.indexOf(u8, text[search_start..], "<think>")) |rel_idx| {
        const idx = search_start + rel_idx;
        const prefix = std.mem.trim(u8, text[0..idx], " \t\r\n");
        if (prefix.len > 0) return idx;
    }
    return null;
}

fn trimUnexpectedThinkingTail(text: []const u8) []const u8 {
    if (findUnexpectedThinkingTailStart(text)) |idx| {
        const trimmed = std.mem.trimRight(u8, text[0..idx], " \t\r\n");
        if (trimmed.len > 0) {
            return trimmed;
        }
    }
    return text;
}

/// Detect a repeated phrase appearing 3+ times in text.
/// Picks candidate phrases from sentence-boundary positions (after ". ")
/// and checks if they repeat 3+ times. Returns the index of the first repeat region.
fn findRepeatedPhraseLoop(text: []const u8) ?usize {
    if (text.len < 80) return null;
    // Collect candidate start positions: after ". " or start of text
    var starts: [64]usize = undefined;
    var n_starts: usize = 0;
    starts[0] = 0;
    n_starts = 1;
    var si: usize = 0;
    while (si + 1 < text.len and n_starts < starts.len) : (si += 1) {
        if (text[si] == '.' and text[si + 1] == ' ') {
            starts[n_starts] = si + 2;
            n_starts += 1;
        }
    }
    // For each candidate start, try phrase lengths 20..60
    for (starts[0..n_starts]) |start| {
        if (start + 20 > text.len) continue;
        var plen: usize = 20;
        while (plen <= @min(60, text.len - start)) : (plen += 5) {
            const phrase = text[start .. start + plen];
            var count: usize = 1;
            var search_from = start + plen;
            var second_pos: usize = 0;
            while (search_from + plen <= text.len) {
                if (std.mem.indexOf(u8, text[search_from..], phrase)) |rel| {
                    count += 1;
                    if (count == 2) second_pos = search_from + rel;
                    if (count >= 3) return second_pos;
                    search_from = search_from + rel + plen;
                } else break;
            }
        }
    }
    return null;
}

fn findStreamingStopStart(text: []const u8) ?usize {
    var first: ?usize = findFirstStop(text, chat_stop_strs[0..]);
    if (findUnexpectedThinkingTailStart(text)) |idx| {
        if (first == null or idx < first.?) first = idx;
    }
    if (findLeakedReasoningStart(text)) |idx| {
        if (first == null or idx < first.?) first = idx;
    }
    if (findRepeatedPhraseLoop(text)) |idx| {
        if (first == null or idx < first.?) first = idx;
    }
    return first;
}

fn sanitizeAnswerTail(text: []const u8) []const u8 {
    return trimRestartedAnswer(trimLeakedNoThinkingOutput(trimUnexpectedThinkingTail(trimTrailingChatArtifacts(text))));
}

fn sanitizeStreamingAnswerTail(text: []const u8) []const u8 {
    return trimRestartedAnswer(trimLeakedNoThinkingOutput(trimUnexpectedThinkingTail(text)));
}

fn sanitizeThinkingOutput(text: []const u8, buf: []u8) ![]const u8 {
    if (!std.mem.startsWith(u8, text, "<think>")) return sanitizeAnswerTail(text);
    const close_idx = std.mem.indexOf(u8, text, "</think>") orelse return text;
    const reasoning_end = close_idx + "</think>".len;
    const answer = sanitizeAnswerTail(text[reasoning_end..]);
    const joiner = if (answer.len > 0 and !std.mem.startsWith(u8, answer, "\n")) "\n" else "";
    const total_len = reasoning_end + joiner.len + answer.len;
    if (total_len > buf.len) return error.BufferTooSmall;
    @memcpy(buf[0..reasoning_end], text[0..reasoning_end]);
    var pos = reasoning_end;
    if (joiner.len > 0) {
        @memcpy(buf[pos .. pos + joiner.len], joiner);
        pos += joiner.len;
    }
    if (answer.len > 0) {
        @memcpy(buf[pos .. pos + answer.len], answer);
        pos += answer.len;
    }
    return buf[0..pos];
}

fn sanitizeStreamingThinkingOutput(text: []const u8, buf: []u8) ![]const u8 {
    if (!std.mem.startsWith(u8, text, "<think>")) return sanitizeStreamingAnswerTail(text);
    const close_idx = std.mem.indexOf(u8, text, "</think>") orelse return text;
    const reasoning_end = close_idx + "</think>".len;
    const answer = sanitizeStreamingAnswerTail(text[reasoning_end..]);
    const joiner = if (answer.len > 0 and !std.mem.startsWith(u8, answer, "\n")) "\n" else "";
    const total_len = reasoning_end + joiner.len + answer.len;
    if (total_len > buf.len) return error.BufferTooSmall;
    @memcpy(buf[0..reasoning_end], text[0..reasoning_end]);
    var pos = reasoning_end;
    if (joiner.len > 0) {
        @memcpy(buf[pos .. pos + joiner.len], joiner);
        pos += joiner.len;
    }
    if (answer.len > 0) {
        @memcpy(buf[pos .. pos + answer.len], answer);
        pos += answer.len;
    }
    return buf[0..pos];
}

fn findRestartedAnswerStart(text: []const u8) ?usize {
    if (text.len < 160) return null;
    const prefix_len: usize = @min(text.len, @as(usize, 96));
    if (prefix_len < 48) return null;
    const prefix = text[0..prefix_len];
    var search_from: usize = prefix_len + @as(usize, 64);
    while (search_from + prefix_len <= text.len) {
        const idx = std.mem.indexOfPos(u8, text, search_from, prefix) orelse return null;
        const at_line_start = idx > 0 and text[idx - 1] == '\n';
        const at_paragraph_start = idx > 1 and text[idx - 2] == '\n' and text[idx - 1] == '\n';
        if (at_line_start or at_paragraph_start) return idx;
        search_from = idx + 1;
    }
    return null;
}

fn trimRestartedAnswer(text: []const u8) []const u8 {
    if (findRestartedAnswerStart(text)) |idx| {
        return std.mem.trimRight(u8, text[0..idx], " \t\r\n");
    }
    return text;
}

fn sanitizeAssistantHistoryContent(text: []const u8) []const u8 {
    if (std.mem.startsWith(u8, text, empty_thinking_prefix)) {
        const body = sanitizeAnswerTail(text[empty_thinking_prefix.len..]);
        return if (body.len == 0) text else std.mem.trimRight(u8, text[0 .. empty_thinking_prefix.len + body.len], " \t\r\n");
    }
    return sanitizeAnswerTail(text);
}

fn trimLeadingStandaloneQuote(text: []const u8) []const u8 {
    var s = std.mem.trimLeft(u8, text, " \t\r\n");
    if (s.len == 0 or s[0] != '"') return text;
    var i: usize = 1;
    var saw_newline = false;
    while (i < s.len) : (i += 1) {
        const c = s[i];
        if (c == '\n' or c == '\r') {
            saw_newline = true;
            continue;
        }
        if (c == ' ' or c == '\t') continue;
        break;
    }
    if (!saw_newline) return text;
    return std.mem.trimLeft(u8, s[i..], " \t\r\n");
}

fn supportsEnabledThinking(tokenizer: *const tokenizer_mod.Tokenizer, enable_thinking: ?bool) bool {
    return tokenizer.supportsThinkingToggle() and (enable_thinking orelse false);
}

fn prefixThinkingEnvelope(text: []const u8, enabled: bool, buf: []u8) ![]const u8 {
    if (!enabled or std.mem.startsWith(u8, text, "<think>")) return text;
    if (thinking_prefix.len + text.len > buf.len) return error.BufferTooSmall;
    @memcpy(buf[0..thinking_prefix.len], thinking_prefix);
    @memcpy(buf[thinking_prefix.len .. thinking_prefix.len + text.len], text);
    return buf[0 .. thinking_prefix.len + text.len];
}

fn transportAssistantContent(tokenizer: *const tokenizer_mod.Tokenizer, text: []const u8, thinking_enabled: bool, buf: []u8) ![]const u8 {
    if (thinking_enabled or !tokenizer.supportsThinkingToggle() or std.mem.startsWith(u8, text, "<think>")) return text;
    if (empty_thinking_prefix.len + text.len > buf.len) return error.BufferTooSmall;
    @memcpy(buf[0..empty_thinking_prefix.len], empty_thinking_prefix);
    @memcpy(buf[empty_thinking_prefix.len .. empty_thinking_prefix.len + text.len], text);
    return buf[0 .. empty_thinking_prefix.len + text.len];
}

fn assistantAnswerForHistory(text: []const u8) []const u8 {
    const sanitized = sanitizeAssistantHistoryContent(text);
    if (std.mem.indexOf(u8, sanitized, "</think>")) |idx| {
        const answer = std.mem.trim(u8, sanitized[idx + "</think>".len ..], " \t\r\n");
        if (answer.len > 0) return compactHistoryAnswer(answer);
    }
    return compactHistoryAnswer(sanitized);
}

fn historyAssistantContent(
    tokenizer: *const tokenizer_mod.Tokenizer,
    text: []const u8,
    transport_buf: []u8,
) ![]const u8 {
    const answer = assistantAnswerForHistory(text);
    return transportAssistantContent(tokenizer, answer, false, transport_buf);
}

fn compactHistoryAnswer(answer: []const u8) []const u8 {
    const trimmed = std.mem.trim(u8, answer, " \t\r\n");
    if (trimmed.len <= chat_history_answer_limit_bytes) return trimmed;

    var end = @min(chat_history_answer_limit_bytes, trimmed.len);
    if (end < trimmed.len) {
        while (end > 0 and (trimmed[end] & 0xC0) == 0x80) : (end -= 1) {}
    }
    if (end == 0) return trimmed[0..0];

    const floor = chat_history_answer_limit_bytes * 3 / 5;
    var cut = end;
    var i = end;
    while (i > floor) : (i -= 1) {
        const c = trimmed[i - 1];
        if (c == '\n' or c == '.' or c == '!' or c == '?' or c == ' ') {
            cut = i;
            break;
        }
    }
    return std.mem.trimRight(u8, trimmed[0..cut], " \t\r\n");
}

fn hasDanglingTrailingQuote(text: []const u8) bool {
    const trimmed = std.mem.trimRight(u8, text, " \t\r\n");
    if (trimmed.len == 0 or trimmed[trimmed.len - 1] != '"') return false;
    var body = trimmed[0 .. trimmed.len - 1];
    while (std.mem.endsWith(u8, body, utf8_replacement)) {
        body = body[0 .. body.len - utf8_replacement.len];
    }
    body = std.mem.trimRight(u8, body, " \t\r\n");
    if (body.len == 0) return false;
    var quote_count: usize = 0;
    for (body) |c| {
        if (c == '"') quote_count += 1;
    }
    return quote_count % 2 == 0;
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

fn trimDanglingHeading(text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trimRight(u8, text, " \t\r\n");
    if (trimmed.len == 0) return null;
    const line_start = (std.mem.lastIndexOfScalar(u8, trimmed, '\n') orelse return checkHeading(trimmed, 0));
    return checkHeading(trimmed, line_start + 1);
}

fn checkHeading(trimmed: []const u8, start: usize) ?[]const u8 {
    const line = std.mem.trim(u8, trimmed[start..], " \t\r\n");
    if (line.len == 0 or line.len > 4) return null;
    for (line) |c| {
        if (c != '#') return null;
    }
    if (start == 0) return "";
    return std.mem.trimRight(u8, trimmed[0 .. start - 1], " \t\r\n");
}

fn handleChatCompletions(
    conn: *http.Connection,
    manager: *model_manager_mod.ModelManager,
    server_state: *ServerState,
    body: []const u8,
    allocator: std.mem.Allocator,
) !void {
    var parsed = parseChatRequest(allocator, body) catch {
        try conn.sendError(400, "invalid_request_error", "Invalid JSON in request body");
        return;
    };
    defer parsed.deinit();

    if (countValidChatMessages(parsed.parsed.value.messages) == 0 or parsed.roles.len == 0 or parsed.contents.len == 0) {
        try conn.sendError(400, "invalid_request_error", "Field 'messages' is required");
        return;
    }

    var generation_guard = GenerationGuard.acquire(server_state);
    defer generation_guard.release();
    const resources = manager.currentResources() orelse {
        try conn.sendError(503, "service_unavailable", "No model is currently loaded");
        return;
    };
    const engine = &resources.engine;
    const tokenizer = &resources.tokenizer;
    const model_name = resources.display_name;

    // If the catalog marks this model's thinking as unstable, force-disable thinking
    // and skip the thinking template entirely (no empty <think></think> block).
    const skip_thinking_template = if (resources.managed_id) |mid|
        if (catalog_mod.find(mid)) |cat_entry| !cat_entry.thinking_stable else false
    else
        false;
    if (skip_thinking_template) {
        parsed.enable_thinking = null;
    }
    const sampling = forward_mod.SamplingParams{
        .temperature = if (parsed.temperature <= 0.0001) 0.0 else std.math.clamp(parsed.temperature, 0.0, 2.0),
        .top_p = std.math.clamp(parsed.top_p, 0.0, 1.0),
        .repetition_penalty = if (parsed.temperature > 0.0001 or parsed.top_p < 0.9999) 1.08 else 1.0,
        .top_k = 64,
    };
    const previous_logits_readback = engine.logits_readback_enabled;
    if (sampling.requiresLogitsReadback() and !previous_logits_readback) {
        engine.enableLogitsReadback();
    }
    defer engine.logits_readback_enabled = previous_logits_readback;

    const prompt_capacity = estimateChatPromptBytes(parsed.roles, parsed.contents, supportsEnabledThinking(tokenizer, parsed.enable_thinking));
    const prompt_buf = allocator.alloc(u8, prompt_capacity) catch {
        try conn.sendError(500, "internal_error", "Prompt allocation failed");
        return;
    };
    defer allocator.free(prompt_buf);

    const prompt = buildChatPrompt(tokenizer, parsed.roles, parsed.contents, parsed.enable_thinking, skip_thinking_template, prompt_buf) catch |err| {
        if (err == error.BufferTooSmall) {
            try conn.sendError(400, "invalid_request_error", "Prompt too long");
            return;
        }
        try conn.sendError(500, "internal_error", "Prompt formatting failed");
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
    defer server_state.clearActiveContext();
    if (prompt_tokens.len == 0) {
        try conn.sendError(500, "internal_error", "Tokenization produced no prompt tokens");
        return;
    }

    const ts = @divTrunc(std.time.timestamp(), 1);
    const max_tokens = parsed.max_tokens;
    const req_id = "chatcmpl-zinc0001"; // TODO: T013 unique IDs
    const thinking_enabled = supportsEnabledThinking(tokenizer, parsed.enable_thinking);
    const seed_ns: i128 = std.time.nanoTimestamp();
    const seed_bits: u128 = @bitCast(seed_ns);
    var prng = std.Random.DefaultPrng.init(@truncate(seed_bits));
    const random = prng.random();
    const cacheable_session = parsed.session_id.len > 0;

    var state = forward_mod.DecodeState.init(allocator);
    defer state.deinit();

    var processed_generated_tokens: std.ArrayList(u32) = .{};
    defer processed_generated_tokens.deinit(allocator);

    var cache_assistant_text: ?[]u8 = null;
    defer if (cache_assistant_text) |text| allocator.free(text);
    defer {
        if (cacheable_session) {
            if (cache_assistant_text) |assistant_text| {
                warmChatReuseCache(
                    server_state,
                    resources,
                    tokenizer,
                    engine,
                    thinking_enabled,
                    parsed.session_id,
                    parsed.roles,
                    parsed.contents,
                    assistant_text,
                    &state,
                    prompt_tokens,
                    processed_generated_tokens.items,
                    allocator,
                ) catch |err| {
                    log.info("chat cache disabled: {s}", .{@errorName(err)});
                    server_state.clearChatReuseSession(parsed.session_id);
                };
            } else {
                server_state.clearChatReuseSession(parsed.session_id);
            }
        }
    }

    if (parsed.stream) {
        conn.sendSseStart() catch return;

        {
            var chunk_buf: [1024]u8 = undefined;
            const chunk = std.fmt.bufPrint(&chunk_buf,
                \\{{"id":"{s}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"role":"assistant"}},"finish_reason":null}}]}}
            , .{ req_id, ts, model_name }) catch return;
            conn.writeSseEvent(chunk) catch return;
        }
        if (thinking_enabled) {
            streamText(conn, thinking_prefix, req_id, ts, model_name) catch return;
        }
        if (conn.isPeerClosed()) return;
    }

    const reused_prefix_len = if (cacheable_session)
        server_state.chat_reuse_cache.matchingPrefixLen(parsed.session_id, resources.model_path, prompt_tokens, std.time.nanoTimestamp())
    else
        0;
    if (reused_prefix_len > 0) {
        state.position = @intCast(reused_prefix_len);
        if (reused_prefix_len < prompt_tokens.len) {
            engine.prefillBatch(&state, prompt_tokens[reused_prefix_len..]) catch {
                server_state.clearChatReuseSession(parsed.session_id);
                if (parsed.stream) {
                    conn.writeSseDone() catch {};
                } else {
                    try conn.sendError(500, "internal_error", "Prefill failed");
                }
                return;
            };
        }
        log.info("chat cache hit: session={s} reused={d} appended={d}", .{
            parsed.session_id,
            reused_prefix_len,
            prompt_tokens.len - reused_prefix_len,
        });
    } else {
        if (parsed.session_id.len > 0) server_state.clearChatReuseSession(parsed.session_id);
        engine.prefillBatch(&state, prompt_tokens) catch {
            if (parsed.stream) {
                conn.writeSseDone() catch {};
            } else {
                try conn.sendError(500, "internal_error", "Prefill failed");
            }
            return;
        };
    }
    server_state.setActiveContextTokens(state.position);

    if (parsed.stream) {
        // Decode loop with buffered stop detection.
        // Tokens are buffered and only sent once we confirm they're not part of <|im_end|>.
        const eos = tokenizer.eosId();
        const stop_strs = chat_stop_strs[0..];
        var pending_tokens: [16]u32 = undefined; // tokens waiting to be sent
        var pending_count: usize = 0;
        var gen_text_buf: [32768]u8 = undefined; // accumulated decoded text for stop check
        var gen_text_len: usize = 0;
        var sent_text_len: usize = 0; // how much of gen_text has been confirmed safe to send
        var stopped = false;
        var finish_reason: FinishReason = .stop;

        if (max_tokens > 0) {
            var prev_token = engine.sample(&state, sampling, random);
            var generated: u32 = 0;

            while (generated < max_tokens and prev_token != eos and !stopped) {
                if (conn.isPeerClosed()) return;

                // Accumulate this token's decoded text
                var dec_buf: [256]u8 = undefined;
                const tok_text = tokenizer.decodeToken(prev_token, &dec_buf);
                if (isReplacementArtifact(tok_text)) {
                    if (generated < max_tokens) {
                        if (conn.isPeerClosed()) return;
                        engine.decodeStep(&state, prev_token, true) catch break;
                        processed_generated_tokens.append(allocator, prev_token) catch {};
                        server_state.setActiveContextTokens(state.position);
                        if (conn.isPeerClosed()) return;
                        prev_token = engine.sample(&state, sampling, random);
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

                // Check for explicit chat stops, reopened think blocks, and leaked prompt-analysis tails.
                if (findStreamingStopStart(gen_text_buf[0..gen_text_len])) |stop_idx| {
                    gen_text_len = stop_idx;
                    const pending_text = gen_text_buf[sent_text_len..gen_text_len];
                    const cleaned_pending = trimTrailingChatArtifacts(pending_text);
                    gen_text_len = sent_text_len + cleaned_pending.len;
                    if (cleaned_pending.len > 0) {
                        streamText(conn, cleaned_pending, req_id, ts, model_name) catch return;
                    }
                    sent_text_len = gen_text_len;
                    pending_count = 0;
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

                generated += 1;

                // Generate next token
                if (generated < max_tokens) {
                    if (conn.isPeerClosed()) return;
                    engine.decodeStep(&state, prev_token, true) catch break;
                    processed_generated_tokens.append(allocator, prev_token) catch {};
                    server_state.setActiveContextTokens(state.position);
                    if (conn.isPeerClosed()) return;
                    prev_token = engine.sample(&state, sampling, random);
                } else break;
            }

            if (!stopped and prev_token != eos and generated >= max_tokens) {
                finish_reason = .length;
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
                \\{{"id":"{s}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{}},"finish_reason":"{s}"}}]}}
            , .{ req_id, ts, model_name, @tagName(finish_reason) }) catch "";
            conn.writeSseEvent(chunk) catch return;
        }

        conn.writeSseDone() catch return;
        if (cacheable_session) {
            const trimmed_stream_text = sanitizeAssistantHistoryContent(gen_text_buf[0..gen_text_len]);
            var transport_buf: [32768]u8 = undefined;
            const transport_text = historyAssistantContent(tokenizer, trimmed_stream_text, &transport_buf) catch trimmed_stream_text;
            cache_assistant_text = allocator.dupe(u8, transport_text) catch null;
        }
    } else {
        // Non-streaming: use same prefill+decode loop with stop detection
        var text_buf: std.ArrayList(u8) = .{};
        defer text_buf.deinit(allocator);
        var ns_gen: u32 = 0;
        const ns_eos = tokenizer.eosId();
        const ns_stops = chat_stop_strs[0..];
        var finish_reason: FinishReason = .stop;
        if (max_tokens > 0) {
            var prev = engine.sample(&state, sampling, random);
            while (ns_gen < max_tokens and prev != ns_eos) {
                var decode_buf2: [256]u8 = undefined;
                const tok_utf8 = tokenizer.decodeToken(prev, &decode_buf2);
                if (isReplacementArtifact(tok_utf8)) {
                    engine.decodeStep(&state, prev, true) catch break;
                    processed_generated_tokens.append(allocator, prev) catch {};
                    server_state.setActiveContextTokens(state.position);
                    prev = engine.sample(&state, sampling, random);
                    continue;
                }
                text_buf.appendSlice(allocator, tok_utf8) catch break;
                ns_gen += 1;
                const hit = if (findFirstStop(text_buf.items, ns_stops)) |pos| blk: {
                    text_buf.shrinkRetainingCapacity(pos);
                    break :blk true;
                } else false;
                if (hit) break;
                if (ns_gen >= max_tokens) break;
                engine.decodeStep(&state, prev, true) catch break;
                processed_generated_tokens.append(allocator, prev) catch {};
                server_state.setActiveContextTokens(state.position);
                prev = engine.sample(&state, sampling, random);
            }
            if (prev != ns_eos and ns_gen >= max_tokens and findFirstStop(text_buf.items, ns_stops) == null) {
                finish_reason = .length;
            }
        }

        // Escape the full text for JSON
        const trimmed_text = if (thinking_enabled)
            trimTrailingChatArtifacts(text_buf.items)
        else
            sanitizeAssistantHistoryContent(text_buf.items);
        var thinking_buf: [16384]u8 = undefined;
        const prefixed_text = prefixThinkingEnvelope(trimmed_text, thinking_enabled, &thinking_buf) catch trimmed_text;
        var sanitized_thinking_buf: [16384]u8 = undefined;
        const response_text = if (thinking_enabled)
            sanitizeThinkingOutput(prefixed_text, &sanitized_thinking_buf) catch prefixed_text
        else
            prefixed_text;
        var escaped_buf: [16384]u8 = undefined;
        const escaped_text = jsonEscape(response_text, &escaped_buf);

        var resp_buf: [32768]u8 = undefined;
        const resp = std.fmt.bufPrint(&resp_buf,
            \\{{"id":"{s}","object":"chat.completion","created":{d},"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"}},"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
        , .{
            req_id,                     ts,                model_name,
            escaped_text,               @tagName(finish_reason), prompt_tokens.len, ns_gen,
            prompt_tokens.len + ns_gen,
        }) catch {
            try conn.sendError(500, "internal_error", "Response too large");
            return;
        };
        try conn.sendJson(200, resp);
        if (cacheable_session) {
            var transport_buf: [32768]u8 = undefined;
            const transport_text = historyAssistantContent(tokenizer, response_text, &transport_buf) catch response_text;
            cache_assistant_text = allocator.dupe(u8, transport_text) catch null;
        }
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
    const parsed = parseJsonFields(body) catch {
        try conn.sendError(400, "invalid_request_error", "Invalid JSON");
        return;
    };

    if (parsed.prompt_text.len == 0) {
        try conn.sendError(400, "invalid_request_error", "Field 'prompt' is required");
        return;
    }

    var generation_guard = GenerationGuard.acquire(server_state);
    defer generation_guard.release();
    const resources = manager.currentResources() orelse {
        try conn.sendError(503, "service_unavailable", "No model is currently loaded");
        return;
    };
    const tokenizer = &resources.tokenizer;
    const engine = &resources.engine;
    const model_name = resources.display_name;

    // Tokenize raw prompt (no chat template)
    const prompt_tokens = tokenizer.encodePrompt(parsed.prompt_text, allocator) catch {
        try conn.sendError(500, "internal_error", "Tokenization failed");
        return;
    };
    defer allocator.free(prompt_tokens);
    defer server_state.clearActiveContext();
    if (prompt_tokens.len == 0) {
        try conn.sendError(500, "internal_error", "Tokenization produced no prompt tokens");
        return;
    }
    server_state.setActiveContextTokens(@intCast(@min(prompt_tokens.len, std.math.maxInt(u32))));

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

/// Minimal JSON field extraction (no full parser — just find key fields).
const ParsedRequest = struct {
    model_id: []const u8,
    messages_content: []const u8, // last user message content
    prompt_text: []const u8, // raw prompt for /v1/completions
    max_tokens: u32,
    stream: bool,
    force: bool,
    temperature: f32,
    enable_thinking: ?bool,
};

fn parseJsonFields(body: []const u8) !ParsedRequest {
    var result = ParsedRequest{
        .model_id = "",
        .messages_content = "",
        .prompt_text = "",
        .max_tokens = 256,
        .stream = false,
        .force = false,
        .temperature = 1.0,
        .enable_thinking = null,
    };

    // Extract "stream":true/false
    if (std.mem.indexOf(u8, body, "\"stream\":true") != null or
        std.mem.indexOf(u8, body, "\"stream\": true") != null)
    {
        result.stream = true;
    }

    if (std.mem.indexOf(u8, body, "\"force\":true") != null or
        std.mem.indexOf(u8, body, "\"force\": true") != null)
    {
        result.force = true;
    }

    if (std.mem.indexOf(u8, body, "\"enable_thinking\":true") != null or
        std.mem.indexOf(u8, body, "\"enable_thinking\": true") != null)
    {
        result.enable_thinking = true;
    } else if (std.mem.indexOf(u8, body, "\"enable_thinking\":false") != null or
        std.mem.indexOf(u8, body, "\"enable_thinking\": false") != null)
    {
        result.enable_thinking = false;
    }

    // Extract "max_tokens":N
    if (std.mem.indexOf(u8, body, "\"max_tokens\":")) |pos| {
        const start = pos + "\"max_tokens\":".len;
        const trimmed = std.mem.trim(u8, body[start..@min(start + 10, body.len)], " ");
        result.max_tokens = std.fmt.parseInt(u32, trimmed[0..findNumEnd(trimmed)], 10) catch 256;
    } else if (std.mem.indexOf(u8, body, "\"max_tokens\": ")) |pos| {
        const start = pos + "\"max_tokens\": ".len;
        result.max_tokens = std.fmt.parseInt(u32, body[start..@min(start + 10, body.len)][0..findNumEnd(body[start..@min(start + 10, body.len)])], 10) catch 256;
    }

    // Extract last "content":"..." from messages
    if (std.mem.lastIndexOf(u8, body, "\"content\":\"")) |pos| {
        const start = pos + "\"content\":\"".len;
        if (findStringEnd(body[start..])) |end| {
            result.messages_content = body[start .. start + end];
        }
    } else if (std.mem.lastIndexOf(u8, body, "\"content\": \"")) |pos| {
        const start = pos + "\"content\": \"".len;
        if (findStringEnd(body[start..])) |end| {
            result.messages_content = body[start .. start + end];
        }
    }

    // Extract "prompt":"..."
    if (std.mem.indexOf(u8, body, "\"prompt\":\"")) |pos| {
        const start = pos + "\"prompt\":\"".len;
        if (findStringEnd(body[start..])) |end| {
            result.prompt_text = body[start .. start + end];
        }
    } else if (std.mem.indexOf(u8, body, "\"prompt\": \"")) |pos| {
        const start = pos + "\"prompt\": \"".len;
        if (findStringEnd(body[start..])) |end| {
            result.prompt_text = body[start .. start + end];
        }
    }

    // Extract "model":"..."
    if (std.mem.indexOf(u8, body, "\"model\":\"")) |pos| {
        const start = pos + "\"model\":\"".len;
        if (findStringEnd(body[start..])) |end| {
            result.model_id = body[start .. start + end];
        }
    } else if (std.mem.indexOf(u8, body, "\"model\": \"")) |pos| {
        const start = pos + "\"model\": \"".len;
        if (findStringEnd(body[start..])) |end| {
            result.model_id = body[start .. start + end];
        }
    }

    return result;
}

fn findNumEnd(s: []const u8) usize {
    for (s, 0..) |c, i| {
        if (c < '0' or c > '9') return i;
    }
    return s.len;
}

fn findStringEnd(s: []const u8) ?usize {
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (s[i] == '\\') {
            i += 1;
            continue;
        } // skip escaped chars
        if (s[i] == '"') return i;
    }
    return null;
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
    var escaped_buf: [8192]u8 = undefined;
    const escaped = jsonEscape(text, &escaped_buf);
    var chunk_buf: [16384]u8 = undefined;
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

test "parseJsonFields extracts stream flag" {
    const body = "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}";
    const parsed = try parseJsonFields(body);
    try std.testing.expect(parsed.stream);
    try std.testing.expectEqualStrings("hello", parsed.messages_content);
    try std.testing.expect(parsed.enable_thinking == null);
}

test "parseJsonFields extracts force flag" {
    const parsed = try parseJsonFields("{\"model\":\"qwen\",\"force\": true}");
    try std.testing.expect(parsed.force);
}

test "parseJsonFields extracts max_tokens" {
    const body = "{\"model\":\"qwen\",\"prompt\":\"test\",\"max_tokens\":128}";
    const parsed = try parseJsonFields(body);
    try std.testing.expectEqual(@as(u32, 128), parsed.max_tokens);
    try std.testing.expectEqualStrings("test", parsed.prompt_text);
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

test "parseJsonFields defaults when fields missing" {
    const body = "{\"model\":\"qwen\"}";
    const parsed = try parseJsonFields(body);
    try std.testing.expect(!parsed.stream);
    try std.testing.expectEqual(@as(u32, 256), parsed.max_tokens);
    try std.testing.expectEqualStrings("", parsed.messages_content);
    try std.testing.expectEqualStrings("", parsed.prompt_text);
    try std.testing.expect(parsed.enable_thinking == null);
}

test "parseJsonFields stream false explicit" {
    const body = "{\"model\":\"qwen\",\"stream\":false}";
    const parsed = try parseJsonFields(body);
    try std.testing.expect(!parsed.stream);
}

test "parseJsonFields extracts enable_thinking flag" {
    const enabled = try parseJsonFields("{\"enable_thinking\":true}");
    try std.testing.expectEqual(@as(?bool, true), enabled.enable_thinking);

    const disabled = try parseJsonFields("{\"enable_thinking\": false}");
    try std.testing.expectEqual(@as(?bool, false), disabled.enable_thinking);
}

test "parseJsonFields extracts content with spaces" {
    const body = "{\"model\":\"q\",\"messages\":[{\"role\":\"user\",\"content\": \"hello world\"}],\"stream\": true}";
    const parsed = try parseJsonFields(body);
    try std.testing.expect(parsed.stream);
    try std.testing.expectEqualStrings("hello world", parsed.messages_content);
}

test "parseJsonFields max_tokens with spaces" {
    const body = "{\"max_tokens\": 64}";
    const parsed = try parseJsonFields(body);
    try std.testing.expectEqual(@as(u32, 64), parsed.max_tokens);
}

test "findStringEnd handles escaped quotes" {
    // Input after opening quote: hello \"inner\" end"rest
    // Escaped \" at positions 6-7 and 14-15, real " at position 19
    const s = "hello \\\"inner\\\" end\"rest";
    const end = findStringEnd(s);
    try std.testing.expectEqual(@as(?usize, 19), end);
}

test "findNumEnd extracts digits" {
    try std.testing.expectEqual(@as(usize, 3), findNumEnd("123abc"));
    try std.testing.expectEqual(@as(usize, 0), findNumEnd("abc"));
    try std.testing.expectEqual(@as(usize, 5), findNumEnd("99999"));
}

test "parseJsonFields handles multiline content" {
    const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"line1\\nline2\"}]}";
    const parsed = try parseJsonFields(body);
    try std.testing.expectEqualStrings("line1\\nline2", parsed.messages_content);
}

test "parseJsonFields handles multiple messages picks last content" {
    const body =
        \\{"messages":[{"role":"system","content":"sys"},{"role":"user","content":"usr"}]}
    ;
    const parsed = try parseJsonFields(body);
    // lastIndexOf should find the last "content" which is "usr"
    try std.testing.expectEqualStrings("usr", parsed.messages_content);
}

test "parseJsonFields max_tokens large value" {
    const body = "{\"max_tokens\":4096}";
    const parsed = try parseJsonFields(body);
    try std.testing.expectEqual(@as(u32, 4096), parsed.max_tokens);
}

test "parseJsonFields prompt with special chars" {
    const body = "{\"prompt\":\"What is 2+2?\"}";
    const parsed = try parseJsonFields(body);
    try std.testing.expectEqualStrings("What is 2+2?", parsed.prompt_text);
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

test "findStringEnd no closing quote returns null" {
    try std.testing.expectEqual(@as(?usize, null), findStringEnd("no close"));
}

test "findStringEnd immediate close" {
    try std.testing.expectEqual(@as(?usize, 0), findStringEnd("\"rest"));
}

test "buildChatPrompt uses tokenizer chat template helper" {
    var tok = makeTestTokenizer(null);
    defer tok.token_to_id.deinit();

    const roles = [_][]const u8{"user"};
    const contents = [_][]const u8{"hello"};
    var buf: [512]u8 = undefined;
    const prompt = try buildChatPrompt(&tok, &roles, &contents, null, false, &buf);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>system\n") == null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "Do not output labels like 'Thinking Process:'") == null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>user\nhello<|im_end|>\n") != null);
    try std.testing.expect(std.mem.endsWith(u8, prompt, "<|im_start|>assistant\n"));
}

test "trimTrailingChatArtifacts strips endoftext and replacement junk" {
    const text = "Hello!\n \xEF\xBF\xBD\xEF\xBF\xBD<|endoftext|>\n\n";
    try std.testing.expectEqualStrings("Hello!", trimTrailingChatArtifacts(text));
}

test "trimTrailingChatArtifacts strips unmatched trailing quote after punctuation" {
    try std.testing.expectEqualStrings("Hello! How can I help you today?", trimTrailingChatArtifacts("Hello! How can I help you today?\"\n\n"));
    try std.testing.expectEqualStrings("Hey there! How can I help you today?", trimTrailingChatArtifacts("Hey there! How can I help you today? \xEF\xBF\xBD\xEF\xBF\xBD\"\n"));
    try std.testing.expectEqualStrings("Hey there! How can I help you today? 😊", trimTrailingChatArtifacts("Hey there! How can I help you today? 😊\"\n"));
    try std.testing.expectEqualStrings("\"Paris.\"", trimTrailingChatArtifacts("\"Paris.\""));
}

test "trimTrailingChatArtifacts strips dangling heading markers" {
    try std.testing.expectEqualStrings("Hello", trimTrailingChatArtifacts("Hello\n\n###\n"));
    try std.testing.expectEqualStrings("", trimTrailingChatArtifacts("###\n"));
}

test "trimTrailingChatArtifacts strips dangling list marker" {
    try std.testing.expectEqualStrings(
        "Cons:\n- Ecosystem is less mature",
        trimTrailingChatArtifacts("Cons:\n- Ecosystem is less mature\n-"),
    );
}

test "trimTrailingChatArtifacts strips leading standalone quote before answer text" {
    try std.testing.expectEqualStrings("Vulcan is likely a typo for Vulkan.", trimTrailingChatArtifacts("\"\n\nVulcan is likely a typo for Vulkan."));
}

test "trimLeakedNoThinkingOutput strips self-referential planning suffix" {
    const raw =
        "Kernel development requires:\n" ++
        "- No standard library\n" ++
        "- Direct hardware access\n" ++
        "\n\nThe user is asking about writing kernel programs in Zig.\n" ++
        "Here is the response:\n" ++
        "Kernel development requires:\n" ++
        "- No standard library";
    try std.testing.expectEqualStrings(
        "Kernel development requires:\n- No standard library\n- Direct hardware access",
        trimLeakedNoThinkingOutput(raw),
    );
}

test "sanitizeThinkingOutput strips leaked planning from answer tail" {
    const raw =
        "<think>\nReasoning.\n</think>\n" ++
        "Zig is increasingly being considered for kernel work.\n\n" ++
        "However, looking at the prompt structure, it seems I am generating the next turn.";
    var buf: [512]u8 = undefined;
    const cleaned = try sanitizeThinkingOutput(raw, &buf);
    try std.testing.expectEqualStrings(
        "<think>\nReasoning.\n</think>\nZig is increasingly being considered for kernel work.",
        cleaned,
    );
}

test "sanitizeThinkingOutput strips reopened think block from answer tail" {
    const raw =
        "<think>\nReasoning.\n</think>\n" ++
        "Zig is promising for kernel programming.\n\n" ++
        "<think>\nThinking Process:\n1. Analyze the request.\n";
    var buf: [512]u8 = undefined;
    const cleaned = try sanitizeThinkingOutput(raw, &buf);
    try std.testing.expectEqualStrings(
        "<think>\nReasoning.\n</think>\nZig is promising for kernel programming.",
        cleaned,
    );
}

test "sanitizeStreamingThinkingOutput strips reopened think block from answer tail" {
    const raw =
        "<think>\nReasoning.\n</think>\n" ++
        "Zig is promising for kernel programming.\n\n" ++
        "<think>\nThinking Process:\n1. Analyze the request.\n";
    var buf: [512]u8 = undefined;
    const cleaned = try sanitizeStreamingThinkingOutput(raw, &buf);
    try std.testing.expectEqualStrings(
        "<think>\nReasoning.\n</think>\nZig is promising for kernel programming.",
        cleaned,
    );
}

test "trimUnexpectedThinkingTail strips reopened think block without leading newline" {
    const raw =
        "Overall, Zig has potential for kernel programming." ++
        "<think>\nThinking Process:\n1. Analyze the request.\n";
    try std.testing.expectEqualStrings(
        "Overall, Zig has potential for kernel programming.",
        trimUnexpectedThinkingTail(raw),
    );
}

test "findStreamingStopStart detects leaked prompt-analysis tail" {
    const raw =
        "Overall, while Zig has potential, it is not yet the best choice for production kernel programming." ++
        "<think>\nThinking Process:\n1. Analyze the Request:\n" ++
        "    *   Current State: The assistant has already provided a response in the few-shot example.";
    try std.testing.expect(findStreamingStopStart(raw) != null);
}

test "trimRestartedAnswer strips duplicated restart from opening paragraph" {
    const raw =
        "To write kernel programs in Zig, you need to understand that Zig is primarily designed for user-space applications.\n" ++
        "It also supports low-level systems programming.\n\n" ++
        "To write kernel programs in Zig, you need to understand that Zig is primarily designed for user-space applications.\n" ++
        "Here is a code example:";
    try std.testing.expectEqualStrings(
        "To write kernel programs in Zig, you need to understand that Zig is primarily designed for user-space applications.\nIt also supports low-level systems programming.",
        trimRestartedAnswer(raw),
    );
}

test "sanitizeAssistantHistoryContent strips leaked planning and duplicate restart" {
    const raw =
        "To write kernel programs in Zig, you need to understand that Zig is primarily designed for user-space applications.\n" ++
        "It also supports low-level systems programming.\n\n" ++
        "The user is asking about kernel programs in Zig.\n" ++
        "Here is the response:\n" ++
        "To write kernel programs in Zig, you need to understand that Zig is primarily designed for user-space applications.";
    try std.testing.expectEqualStrings(
        "To write kernel programs in Zig, you need to understand that Zig is primarily designed for user-space applications.\nIt also supports low-level systems programming.",
        sanitizeAssistantHistoryContent(raw),
    );
}

test "isReplacementArtifact detects replacement-only chunks" {
    try std.testing.expect(isReplacementArtifact(" \xEF\xBF\xBD"));
    try std.testing.expect(isReplacementArtifact("\xEF\xBF\xBD\xEF\xBF\xBD"));
    try std.testing.expect(!isReplacementArtifact("Hello \xEF\xBF\xBD"));
}

test "hasDanglingTrailingQuote detects unmatched punctuation quote suffix" {
    try std.testing.expect(hasDanglingTrailingQuote("Hello?\""));
    try std.testing.expect(hasDanglingTrailingQuote("Hello?\"\n\n"));
    try std.testing.expect(hasDanglingTrailingQuote("Hello? \xEF\xBF\xBD\xEF\xBF\xBD\""));
    try std.testing.expect(hasDanglingTrailingQuote("Hello 😊\""));
    try std.testing.expect(!hasDanglingTrailingQuote("\"Paris.\""));
    try std.testing.expect(!hasDanglingTrailingQuote("He said \"hi\""));
}

test "buildChatPrompt uses qwen no-thinking generation suffix when template requests it" {
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

    const roles = [_][]const u8{"user"};
    const contents = [_][]const u8{"hello"};
    var buf: [512]u8 = undefined;
    const prompt = try buildChatPrompt(&tok, &roles, &contents, null, false, &buf);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>user\nhello<|im_end|>\n") != null);
    try std.testing.expect(std.mem.endsWith(u8, prompt, "<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}

test "buildChatPrompt enables thinking when requested" {
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

    const roles = [_][]const u8{"user"};
    const contents = [_][]const u8{"hello"};
    var buf: [512]u8 = undefined;
    const prompt = try buildChatPrompt(&tok, &roles, &contents, true, false, &buf);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>user\nhello<|im_end|>\n") != null);
    try std.testing.expect(std.mem.endsWith(u8, prompt, "<|im_start|>assistant\n<think>\n"));
    try std.testing.expect(std.mem.indexOf(u8, prompt, "</think>") == null);
}

test "parseChatRequest preserves full message history" {
    const body =
        \\{"messages":[{"role":"system","content":"be concise"},{"role":"user","content":"hello"},{"role":"assistant","content":"hi"},{"role":"user","content":"follow up"}],"max_tokens":128,"stream":true,"temperature":0.7,"top_p":0.9,"enable_thinking":true}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 4), parsed.roles.len);
    try std.testing.expectEqualStrings("system", parsed.roles[0]);
    try std.testing.expectEqualStrings("be concise", parsed.contents[0]);
    try std.testing.expectEqualStrings("assistant", parsed.roles[2]);
    try std.testing.expectEqualStrings("hi", parsed.contents[2]);
    try std.testing.expectEqualStrings("user", parsed.roles[3]);
    try std.testing.expectEqualStrings("follow up", parsed.contents[3]);
    try std.testing.expectEqual(@as(u32, 128), parsed.max_tokens);
    try std.testing.expect(parsed.stream);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), parsed.temperature, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), parsed.top_p, 0.0001);
    try std.testing.expectEqual(@as(?bool, true), parsed.enable_thinking);
}

test "parseChatRequest prepends default system guidance when missing" {
    const body =
        \\{"messages":[{"role":"user","content":"tell me how I can do inference on Vulcan + zig"}]}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 2), parsed.roles.len);
    try std.testing.expectEqualStrings("system", parsed.roles[0]);
    try std.testing.expect(std.mem.indexOf(u8, parsed.contents[0], "ambiguous") != null);
    try std.testing.expectEqualStrings("user", parsed.roles[1]);
}

test "parseChatRequest defaults to greedy temperature when omitted" {
    const body =
        \\{"messages":[{"role":"user","content":"hello"}]}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(f32, 0.0), parsed.temperature);
}

test "countValidChatMessages ignores empty entries" {
    const messages = [_]ChatMessage{
        .{ .role = "", .content = "ignored" },
        .{ .role = "user", .content = "" },
        .{ .role = "user", .content = "hello" },
    };
    try std.testing.expectEqual(@as(usize, 1), countValidChatMessages(&messages));
}

test "parseChatRequest leaves empty message array empty before validation" {
    const body = "{}";
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 0), countValidChatMessages(parsed.parsed.value.messages));
    try std.testing.expectEqual(@as(usize, 1), parsed.roles.len);
    try std.testing.expectEqualStrings("system", parsed.roles[0]);
}

test "prefixThinkingEnvelope adds think prefix when enabled" {
    var buf: [128]u8 = undefined;
    const prefixed = try prefixThinkingEnvelope("17 * 24 = 408\n</think>\n408", true, &buf);
    try std.testing.expectEqualStrings("<think>\n17 * 24 = 408\n</think>\n408", prefixed);
}

test "prefixThinkingEnvelope leaves text unchanged when disabled" {
    var buf: [64]u8 = undefined;
    const plain = try prefixThinkingEnvelope("408", false, &buf);
    try std.testing.expectEqualStrings("408", plain);
}

test "transportAssistantContent adds empty think scaffold for non-thinking qwen history" {
    var qwen_tok = makeTestTokenizer(
        \\{%- if add_generation_prompt %}
        \\  {{- '<|im_start|>assistant\n' }}
        \\  {%- if enable_thinking is defined and enable_thinking is true %}
        \\    {{- '<think>\n' }}
        \\  {%- else %}
        \\    {{- '<think>\n\n</think>\n\n' }}
        \\  {%- endif %}
        \\{%- endif %}
    );
    defer qwen_tok.token_to_id.deinit();

    var buf: [128]u8 = undefined;
    const transport = try transportAssistantContent(&qwen_tok, "Kernel code needs explicit resource control.", false, &buf);
    try std.testing.expectEqualStrings("<think>\n\n</think>\n\nKernel code needs explicit resource control.", transport);
}

test "transportAssistantContent leaves already-prefixed text unchanged" {
    var qwen_tok = makeTestTokenizer(
        \\{%- if add_generation_prompt %}
        \\  {{- '<|im_start|>assistant\n' }}
        \\  {%- if enable_thinking is defined and enable_thinking is true %}
        \\    {{- '<think>\n' }}
        \\  {%- else %}
        \\    {{- '<think>\n\n</think>\n\n' }}
        \\  {%- endif %}
        \\{%- endif %}
    );
    defer qwen_tok.token_to_id.deinit();

    var buf: [128]u8 = undefined;
    const transport = try transportAssistantContent(&qwen_tok, "<think>\n\n</think>\n\nKernel code needs explicit resource control.", false, &buf);
    try std.testing.expectEqualStrings("<think>\n\n</think>\n\nKernel code needs explicit resource control.", transport);
}

test "assistantAnswerForHistory drops completed reasoning blocks" {
    const answer = assistantAnswerForHistory("<think>\nReason step\n</think>\nAnswer text.");
    try std.testing.expectEqualStrings("Answer text.", answer);
}

test "assistantAnswerForHistory compacts long answers" {
    const long_answer = "Kernel programming in Zig gives you explicit control. " ** 24;
    const compacted = assistantAnswerForHistory(long_answer);
    try std.testing.expect(compacted.len < long_answer.len);
    try std.testing.expect(std.mem.startsWith(u8, compacted, "Kernel programming in Zig"));
}

test "historyAssistantContent strips streamed reasoning before caching" {
    var qwen_tok = makeTestTokenizer(
        \\{%- if add_generation_prompt %}
        \\  {{- '<|im_start|>assistant\n' }}
        \\  {%- if enable_thinking is defined and enable_thinking is true %}
        \\    {{- '<think>\n' }}
        \\  {%- else %}
        \\    {{- '<think>\n\n</think>\n\n' }}
        \\  {%- endif %}
        \\{%- endif %}
    );
    defer qwen_tok.token_to_id.deinit();

    var transport_buf: [160]u8 = undefined;
    const cached = try historyAssistantContent(
        &qwen_tok,
        "Reason step\n</think>\nAnswer text.",
        &transport_buf,
    );
    try std.testing.expectEqualStrings("<think>\n\n</think>\n\nAnswer text.", cached);
}

test "supportsEnabledThinking requires tokenizer support and request flag" {
    var qwen_tok = makeTestTokenizer(
        \\{%- if add_generation_prompt %}
        \\  {{- '<|im_start|>assistant\n' }}
        \\  {%- if enable_thinking is defined and enable_thinking is true %}
        \\    {{- '<think>\n' }}
        \\  {%- else %}
        \\    {{- '<think>\n\n</think>\n\n' }}
        \\  {%- endif %}
        \\{%- endif %}
    );
    defer qwen_tok.token_to_id.deinit();

    var plain_tok = makeTestTokenizer(null);
    defer plain_tok.token_to_id.deinit();

    try std.testing.expect(supportsEnabledThinking(&qwen_tok, true));
    try std.testing.expect(!supportsEnabledThinking(&qwen_tok, false));
    try std.testing.expect(!supportsEnabledThinking(&plain_tok, true));
}

test "ChatReuseCache stores distinct sessions independently" {
    var cache = ChatReuseCache.init(std.testing.allocator);
    defer cache.deinit();

    try cache.store("session-a", "/tmp/model.gguf", &.{ 1, 2, 3 }, 10);
    try cache.store("session-b", "/tmp/model.gguf", &.{ 4, 5 }, 20);

    try std.testing.expectEqual(@as(usize, 3), cache.matchingPrefixLen("session-a", "/tmp/model.gguf", &.{ 1, 2, 3, 9 }, 30));
    try std.testing.expectEqual(@as(usize, 2), cache.matchingPrefixLen("session-b", "/tmp/model.gguf", &.{ 4, 5, 6 }, 31));
    try std.testing.expectEqual(@as(usize, 0), cache.matchingPrefixLen("session-c", "/tmp/model.gguf", &.{ 1, 2, 3, 9 }, 32));
}

test "ChatReuseCache prunes idle sessions automatically" {
    var cache = ChatReuseCache.init(std.testing.allocator);
    defer cache.deinit();

    try cache.store("stale", "/tmp/model.gguf", &.{ 1, 2, 3 }, 0);
    try cache.store("fresh", "/tmp/model.gguf", &.{ 4, 5, 6 }, chat_reuse_idle_timeout_ns - 1);

    _ = cache.matchingPrefixLen("fresh", "/tmp/model.gguf", &.{ 4, 5, 6, 7 }, chat_reuse_idle_timeout_ns - 1);
    cache.pruneExpired(chat_reuse_idle_timeout_ns + 10);

    try std.testing.expectEqual(@as(usize, 1), cache.count());
    try std.testing.expectEqual(@as(usize, 0), cache.matchingPrefixLen("stale", "/tmp/model.gguf", &.{ 1, 2, 3, 4 }, chat_reuse_idle_timeout_ns + 11));
    try std.testing.expectEqual(@as(usize, 3), cache.matchingPrefixLen("fresh", "/tmp/model.gguf", &.{ 4, 5, 6, 7 }, chat_reuse_idle_timeout_ns + 11));
}

test "ChatReuseCache evicts least recently used session when full" {
    var cache = ChatReuseCache.init(std.testing.allocator);
    defer cache.deinit();

    var session_buf: [64]u8 = undefined;
    var token_pair: [2]u32 = undefined;
    for (0..chat_reuse_max_sessions) |i| {
        const session_id = try std.fmt.bufPrint(&session_buf, "session-{d}", .{i});
        token_pair = .{ @intCast(i), @intCast(i + 100) };
        try cache.store(session_id, "/tmp/model.gguf", token_pair[0..], @intCast(i + 1));
    }
    _ = cache.matchingPrefixLen("session-0", "/tmp/model.gguf", &.{ 0, 100, 999 }, @intCast(chat_reuse_max_sessions + 1));

    const evicted_session = try std.fmt.bufPrint(&session_buf, "session-{d}", .{chat_reuse_max_sessions});
    token_pair = .{ @intCast(chat_reuse_max_sessions), @intCast(chat_reuse_max_sessions + 100) };
    try cache.store(evicted_session, "/tmp/model.gguf", token_pair[0..], @intCast(chat_reuse_max_sessions + 2));

    try std.testing.expectEqual(@as(usize, chat_reuse_max_sessions), cache.count());
    try std.testing.expectEqual(@as(usize, 0), cache.matchingPrefixLen("session-1", "/tmp/model.gguf", &.{ 1, 101, 999 }, @intCast(chat_reuse_max_sessions + 3)));
    try std.testing.expectEqual(@as(usize, 2), cache.matchingPrefixLen("session-0", "/tmp/model.gguf", &.{ 0, 100, 999 }, @intCast(chat_reuse_max_sessions + 4)));
    try std.testing.expectEqual(@as(usize, 2), cache.matchingPrefixLen(evicted_session, "/tmp/model.gguf", &.{ @intCast(chat_reuse_max_sessions), @intCast(chat_reuse_max_sessions + 100), 999 }, @intCast(chat_reuse_max_sessions + 5)));
}

test "ServerState snapshot tracks active queued and uptime" {
    var state = ServerState.init(100);
    _ = state.active_requests.fetchAdd(1, .monotonic);
    _ = state.queued_requests.fetchAdd(2, .monotonic);
    state.setActiveContextTokens(1536);

    const snapshot = state.snapshot(112);
    try std.testing.expectEqual(@as(u32, 1), snapshot.active_requests);
    try std.testing.expectEqual(@as(u32, 2), snapshot.queued_requests);
    try std.testing.expectEqual(@as(u32, 1536), snapshot.active_context_tokens);
    try std.testing.expectEqual(@as(u64, 12), snapshot.uptime_seconds);
}

test "buildHealthJson includes request counts and uptime" {
    var state = ServerState.init(std.time.timestamp() - 5);
    _ = state.active_requests.fetchAdd(1, .monotonic);
    _ = state.queued_requests.fetchAdd(1, .monotonic);
    state.setActiveContextTokens(1024);

    var buf: [1024]u8 = undefined;
    const body = try buildHealthJson(&state, "qwen3.5-35b", .{
        .weights_bytes = 20 * 1024 * 1024 * 1024,
        .runtime_device_local_bytes = 1024 * 1024 * 1024,
        .context_reserved_bytes = 768 * 1024 * 1024,
        .context_capacity_tokens = 4096,
        .context_bytes_per_token = 192 * 1024,
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
    try std.testing.expect(std.mem.indexOf(u8, body, "\"gpu_context_reserved_bytes\":") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"gpu_context_active_bytes\":") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"gpu_context_tokens\":1024") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"gpu_context_capacity_tokens\":4096") != null);
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

test "findRepeatedPhraseLoop detects sentence-level repetition" {
    const looping = "I should cover the main types. I should also mention type safety. I should also mention type safety. I should also mention type safety. I should also mention type safety.";
    try std.testing.expect(findRepeatedPhraseLoop(looping) != null);
}

test "findRepeatedPhraseLoop returns null for normal text" {
    const normal = "Zig is a systems programming language. It features manual memory management. It compiles to native code.";
    try std.testing.expect(findRepeatedPhraseLoop(normal) == null);
}

test "startsWithLeakedReasoning detects meta-commentary at start" {
    try std.testing.expect(startsWithLeakedReasoning("The user is asking about C types."));
    try std.testing.expect(startsWithLeakedReasoning("I need to provide a clear explanation."));
    try std.testing.expect(startsWithLeakedReasoning("  Let me think about this carefully."));
    try std.testing.expect(!startsWithLeakedReasoning("Zig is a modern systems programming language."));
    try std.testing.expect(!startsWithLeakedReasoning("C types include int, float, and char."));
}
