//! Route dispatcher and endpoint handlers for the OpenAI-compatible API.
//! @section API Server
//! Handles /v1/chat/completions, /v1/completions, /v1/models, /health,
//! and a built-in chat UI. Supports both streaming (SSE) and non-streaming responses.
const std = @import("std");
const http = @import("http.zig");
const runtime = @import("runtime.zig");
const forward_mod = runtime.forward_mod;
const catalog_mod = if (runtime.supports_model_management) @import("../model/catalog.zig") else struct {};
const managed_mod = if (runtime.supports_model_management) @import("../model/managed.zig") else struct {};
const model_manager_mod = runtime.model_manager_mod;
const tokenizer_mod = runtime.tokenizer_mod;
const Model = runtime.Model;
const memory_plan = @import("../gpu/memory_plan.zig");
const tool_format = @import("tool_format.zig");

const log = std.log.scoped(.routes);
const inference_timing_log = std.log.scoped(.forward);

/// Cached three-state probe of `ZINC_TOOL_CALLING`. Negative = uncached;
/// 0 = disabled; 1 = enabled. The check happens once per process, lazily.
var tool_calling_state: std.atomic.Value(i8) = .init(-1);

/// Return true if OpenAI-compatible tool calling is enabled. Default on.
/// Set `ZINC_TOOL_CALLING=0` (or `false`) to opt out — useful as a kill
/// switch for clients that misbehave when seeing the new `tool_calls`
/// response shape, or for debugging.
pub fn toolCallingEnabled() bool {
    const cached = tool_calling_state.load(.acquire);
    if (cached >= 0) return cached == 1;
    const enabled = blk: {
        const val = std.process.getEnvVarOwned(std.heap.page_allocator, "ZINC_TOOL_CALLING") catch break :blk true;
        defer std.heap.page_allocator.free(val);
        break :blk !std.mem.eql(u8, val, "0") and !std.mem.eql(u8, val, "false");
    };
    tool_calling_state.store(if (enabled) 1 else 0, .release);
    if (!enabled) log.info("ZINC_TOOL_CALLING=0: OpenAI-compatible tool calling disabled", .{});
    return enabled;
}

/// Test-only override of the tool-calling gate. Sets the cached state directly
/// to bypass the env-var probe. Tests should reset to -1 (uncached) on exit so
/// later tests pick up the real env value.
fn setToolCallingForTest(enabled: bool) void {
    tool_calling_state.store(if (enabled) 1 else 0, .release);
}
fn resetToolCallingForTest() void {
    tool_calling_state.store(-1, .release);
}

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

/// Shared server state tracking active requests, context usage, and generation serialization.
pub const ServerState = struct {
    started_at: i64,
    active_requests: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    queued_requests: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    active_context_tokens: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    generation_mutex: std.Thread.Mutex = .{},
    downloads: DownloadTracker = .{},
    chat_reuse_cache: ChatReuseCache,

    /// Create a new server state anchored to the given UNIX timestamp.
    /// @param started_at UNIX timestamp (seconds) of server startup, stored for uptime calculations.
    /// @returns Initialized `ServerState` with zeroed counters and an empty chat-reuse cache.
    pub fn init(started_at: i64) ServerState {
        return .{
            .started_at = started_at,
            .chat_reuse_cache = ChatReuseCache.init(std.heap.page_allocator),
        };
    }

    /// Release owned resources (chat reuse cache).
    pub fn deinit(self: *ServerState) void {
        self.chat_reuse_cache.deinit();
    }

    /// Return elapsed seconds since the server started.
    /// @param now Current UNIX timestamp (seconds) to compare against `started_at`.
    /// @returns Non-negative elapsed seconds; clamped to 0 if `now` is before `started_at`.
    pub fn uptimeSeconds(self: *const ServerState, now: i64) u64 {
        return @intCast(@max(now - self.started_at, 0));
    }

    /// Atomically capture current request and context counters for the health endpoint.
    /// @param now Current UNIX timestamp (seconds) used to compute uptime in the snapshot.
    /// @returns `HealthSnapshot` with monotonic reads of all live counters and computed uptime.
    pub fn snapshot(self: *const ServerState, now: i64) HealthSnapshot {
        return .{
            .active_requests = self.active_requests.load(.monotonic),
            .queued_requests = self.queued_requests.load(.monotonic),
            .active_context_tokens = self.active_context_tokens.load(.monotonic),
            .uptime_seconds = self.uptimeSeconds(now),
        };
    }

    /// Update the active KV-cache token count reported by the health endpoint.
    /// @param tokens Number of tokens currently occupying the KV cache.
    pub fn setActiveContextTokens(self: *ServerState, tokens: u32) void {
        self.active_context_tokens.store(tokens, .monotonic);
    }

    /// Reset the active context token count to zero.
    pub fn clearActiveContext(self: *ServerState) void {
        self.active_context_tokens.store(0, .monotonic);
    }

    /// Evict all entries from the chat prompt-reuse cache.
    pub fn clearChatReuseCache(self: *ServerState) void {
        self.chat_reuse_cache.clear();
    }

    /// Remove a single session from the chat prompt-reuse cache.
    /// @param session_id Opaque session identifier matching the entry to evict.
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
/// @param manager Model manager used to resolve the active model for inference.
/// @param server_state Shared server metrics and generation serialization lock.
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
        if (comptime runtime.supports_model_management) {
            try handleActivateModel(conn, manager, server_state, request.body, allocator);
        } else {
            try sendUnsupportedModelManagement(conn);
        }
    } else if (request.method == .POST and std.mem.eql(u8, request.path, "/v1/models/pull")) {
        if (comptime runtime.supports_model_management) {
            try handlePullModel(conn, manager, server_state, request.body, allocator);
        } else {
            try sendUnsupportedModelManagement(conn);
        }
    } else if (request.method == .POST and std.mem.eql(u8, request.path, "/v1/models/remove")) {
        if (comptime runtime.supports_model_management) {
            try handleRemoveModel(conn, manager, server_state, request.body);
        } else {
            try sendUnsupportedModelManagement(conn);
        }
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

fn sendUnsupportedModelManagement(conn: *http.Connection) !void {
    try conn.sendError(501, "unsupported_operation", "Model management endpoints are not available on this backend");
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
        const entry_context_length: u32 = if (entry.active) memory_usage.context_capacity_tokens else 0;
        try body.writer(allocator).print(
            \\{{"id":"{s}","object":"model","created":{d},"owned_by":"zinc","context_length":{d},"display_name":"{s}","release_date":"{s}","homepage_url":"{s}","family":"{s}","quantization":"{s}","size_bytes":{d},"installed":{s},"active":{s},"managed":{s},"supported_on_current_gpu":{s},"fits_current_gpu":{s},"required_vram_bytes":{d},"required_vram_with_offload_bytes":{d},"requires_offload_to_fit":{s},"fit_source":"{s}","status":"{s}","supports_thinking_toggle":{s},"downloading":{s},"download_phase":"{s}","downloaded_bytes":{d},"download_total_bytes":{d},"download_error":"{s}"}}
        , .{
            entry.id,
            ts,
            entry_context_length,
            entry.display_name,
            entry.release_date,
            entry.homepage_url,
            entry.family,
            entry.quantization,
            entry.size_bytes,
            if (entry.installed) "true" else "false",
            if (entry.active) "true" else "false",
            if (entry.managed) "true" else "false",
            if (entry.supported_on_current_gpu) "true" else "false",
            if (entry.fits_current_gpu) "true" else "false",
            entry.required_vram_bytes,
            entry.required_vram_with_offload_bytes,
            if (entry.requires_offload_to_fit) "true" else "false",
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
    _manager: *model_manager_mod.ModelManager,
    server_state: *ServerState,
    _body: []const u8,
    _allocator: std.mem.Allocator,
) !void {
    if (comptime !runtime.supports_model_management) {
        try sendUnsupportedModelManagement(conn);
        return;
    }
    var parsed = parseRequestBody(_body, _allocator) catch {
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

    const force_flag = if (parseJsonFields(_body)) |pj| pj.force else |_| false;
    _manager.activateManagedModel(parsed.model_id, true, force_flag) catch |err| {
        const msg = switch (err) {
            error.UnknownManagedModel => "Unknown managed model id",
            error.ModelNotInstalled => "Model is not installed in the local cache",
            error.ModelUnsupportedOnThisGpu => "Model is not validated on the current GPU profile. Retry with \"force\":true to activate it anyway.",
            error.ModelDoesNotFit => "Model does not fit the current GPU memory budget",
            error.GpuAlreadyReserved => "Another zinc process already owns this GPU. Stop it before activating a second model on the same device.",
            else => "Model activation failed",
        };
        const status: u16 = switch (err) {
            error.GpuAlreadyReserved => 409,
            else => 400,
        };
        try conn.sendError(status, "invalid_request_error", msg);
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
    _manager: *model_manager_mod.ModelManager,
    server_state: *ServerState,
    _body: []const u8,
) !void {
    if (comptime !runtime.supports_model_management) {
        try sendUnsupportedModelManagement(conn);
        return;
    }
    const parsed = parseJsonFields(_body) catch {
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

    const result = _manager.removeManagedModel(parsed.model_id, parsed.force) catch |err| {
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
    _manager: *model_manager_mod.ModelManager,
    server_state: *ServerState,
    _body: []const u8,
    _allocator: std.mem.Allocator,
) !void {
    if (comptime !runtime.supports_model_management) {
        try sendUnsupportedModelManagement(conn);
        return;
    }
    var parsed = parseRequestBody(_body, _allocator) catch {
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

    if (!_manager.supportsManagedEntry(entry.*, _allocator)) {
        try conn.sendError(400, "invalid_request_error", "Model is not marked supported for the current GPU profile or VRAM budget");
        return;
    }

    if (managed_mod.isInstalled(parsed.model_id, _allocator)) {
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
const thinking_open_tag = "<think>";
const thinking_close_tag = "</think>";
const harmony_analysis_prefix = "<|channel|>analysis<|message|>";
const harmony_final_prefix = "<|channel|>final<|message|>";
const harmony_stop_strs = [_][]const u8{
    "<|end|>",
    "<|return|>",
    "<|start|>",
    "<|channel|>",
};
const chat_history_answer_limit_bytes: usize = 640;
const default_chat_system_prompt =
    "You are a helpful assistant. Answer directly. Do not show analysis.";

const FinishReason = enum {
    stop,
    length,
    tool_calls,
};

const TimingStats = struct {
    ms: f64,
    tps: f64,
    ms_per_token: f64,
};

fn timingStats(token_count: usize, elapsed_ns: u64) TimingStats {
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    if (token_count == 0 or elapsed_ns == 0) {
        return .{ .ms = elapsed_ms, .tps = 0.0, .ms_per_token = 0.0 };
    }
    const tokens_f = @as(f64, @floatFromInt(token_count));
    const ns_f = @as(f64, @floatFromInt(elapsed_ns));
    return .{
        .ms = elapsed_ms,
        .tps = tokens_f * 1_000_000_000.0 / ns_f,
        .ms_per_token = elapsed_ms / tokens_f,
    };
}

fn elapsedNs(start_ns: i128, end_ns: i128) u64 {
    return if (end_ns > start_ns) @intCast(end_ns - start_ns) else 0;
}

fn logPrefillTiming(token_count: usize, start_ns: i128, end_ns: i128) void {
    const stats = timingStats(token_count, elapsedNs(start_ns, end_ns));
    inference_timing_log.info("Prefill: {d} tokens in {d:.1} ms ({d:.2} tok/s)", .{
        token_count,
        stats.ms,
        stats.tps,
    });
}

fn logGeneratedTiming(token_count: usize, start_ns: i128, end_ns: i128) void {
    const stats = timingStats(token_count, elapsedNs(start_ns, end_ns));
    inference_timing_log.info("Generated {d} tokens in {d:.1} ms — {d:.2} tok/s ({d:.1} ms/tok)", .{
        token_count,
        stats.ms,
        stats.tps,
        stats.ms_per_token,
    });
}

fn completionFinishReason(requested_max_tokens: u32, effective_max_tokens: u32, produced_tokens: usize) FinishReason {
    if (requested_max_tokens > effective_max_tokens and produced_tokens >= @as(usize, effective_max_tokens)) {
        return .length;
    }
    return .stop;
}

/// One block inside an OpenAI-style content array, e.g. `{type: "text", text: "..."}`.
/// Unknown block types (image_url, tool_use, etc.) are accepted but ignored downstream.
const ContentBlock = struct {
    type: []const u8 = "",
    text: []const u8 = "",
};

/// Message content. OpenAI-compatible clients send three shapes:
///   1. plain string             — `"hello"`
///   2. content-block array      — `[{"type":"text","text":"hello"}]`
///   3. null (assistant tool call) — `null`
/// We flatten all three to a single string. Non-text blocks are skipped silently.
const Content = struct {
    text: []const u8 = "",

    pub fn jsonParse(
        allocator: std.mem.Allocator,
        source: anytype,
        options: std.json.ParseOptions,
    ) !Content {
        switch (try source.peekNextTokenType()) {
            .null => {
                _ = try source.next();
                return .{ .text = "" };
            },
            .string => {
                const s = try std.json.innerParse([]const u8, allocator, source, options);
                return .{ .text = s };
            },
            .array_begin => {
                _ = try source.next();
                var pieces: std.ArrayList([]const u8) = .{};
                defer pieces.deinit(allocator);
                while (true) {
                    if ((try source.peekNextTokenType()) == .array_end) {
                        _ = try source.next();
                        break;
                    }
                    const block = try std.json.innerParse(ContentBlock, allocator, source, options);
                    if (std.mem.eql(u8, block.type, "text") and block.text.len > 0) {
                        try pieces.append(allocator, block.text);
                    }
                }
                if (pieces.items.len == 0) return .{ .text = "" };
                if (pieces.items.len == 1) return .{ .text = pieces.items[0] };
                return .{ .text = try std.mem.concat(allocator, u8, pieces.items) };
            },
            else => return error.UnexpectedToken,
        }
    }
};

const HistoricalToolCallFunction = struct {
    name: []const u8 = "",
    arguments: []const u8 = "",
};

const HistoricalToolCall = struct {
    id: []const u8 = "",
    type: []const u8 = "function",
    function: HistoricalToolCallFunction = .{},
};

const RequestToolFunction = struct {
    name: []const u8 = "",
    description: []const u8 = "",
    parameters: std.json.Value = .null,
};

const RequestTool = struct {
    type: []const u8 = "function",
    function: RequestToolFunction = .{},
};

/// Resolution of the OpenAI `tool_choice` request field. `auto` lets the model
/// decide whether to invoke a tool; `required` renders the same tool surface
/// but adds a native prompt-level requirement; `none` suppresses tool rendering
/// even when `tools` is non-empty.
pub const ToolChoice = enum { auto, required, none };

fn parseToolChoice(value: ?std.json.Value) ToolChoice {
    const v = value orelse return .auto;
    switch (v) {
        .string => |s| {
            if (std.mem.eql(u8, s, "none")) return .none;
            if (std.mem.eql(u8, s, "required")) return .required;
            return .auto;
        },
        else => return .auto,
    }
}

fn forcedSingleToolName(tools: []const tool_format.ToolDefinition, tool_choice: ToolChoice) ?[]const u8 {
    if (tool_choice != .required or tools.len != 1) return null;
    return tools[0].name;
}

const JsonStringSlice = struct {
    value: []const u8,
    end: usize,
};

fn skipJsonWhitespace(json: []const u8, pos: *usize) void {
    while (pos.* < json.len) : (pos.* += 1) {
        switch (json[pos.*]) {
            ' ', '\t', '\r', '\n' => {},
            else => return,
        }
    }
}

fn isSafeJsonObjectKeyName(key: []const u8) bool {
    if (key.len == 0) return false;
    for (key) |c| {
        if ((c >= 'a' and c <= 'z') or
            (c >= 'A' and c <= 'Z') or
            (c >= '0' and c <= '9') or
            c == '_' or c == '-' or c == '$')
        {
            continue;
        }
        return false;
    }
    return true;
}

fn parseSimpleJsonStringSlice(json: []const u8, quote_pos: usize) ?JsonStringSlice {
    if (quote_pos >= json.len or json[quote_pos] != '"') return null;
    var i = quote_pos + 1;
    while (i < json.len) : (i += 1) {
        const c = json[i];
        if (c == '\\' or c < 0x20) return null;
        if (c == '"') return .{
            .value = json[quote_pos + 1 .. i],
            .end = i + 1,
        };
    }
    return null;
}

fn firstStringInArrayAfterField(json: []const u8, field_literal: []const u8) ?[]const u8 {
    const field_pos = std.mem.indexOf(u8, json, field_literal) orelse return null;
    var pos = field_pos + field_literal.len;
    const array_rel = std.mem.indexOfScalar(u8, json[pos..], '[') orelse return null;
    pos += array_rel + 1;
    skipJsonWhitespace(json, &pos);
    const parsed = parseSimpleJsonStringSlice(json, pos) orelse return null;
    return if (isSafeJsonObjectKeyName(parsed.value)) parsed.value else null;
}

fn firstObjectKeyAfterField(json: []const u8, field_literal: []const u8) ?[]const u8 {
    const field_pos = std.mem.indexOf(u8, json, field_literal) orelse return null;
    var pos = field_pos + field_literal.len;
    const object_rel = std.mem.indexOfScalar(u8, json[pos..], '{') orelse return null;
    pos += object_rel + 1;
    skipJsonWhitespace(json, &pos);
    const parsed = parseSimpleJsonStringSlice(json, pos) orelse return null;
    return if (isSafeJsonObjectKeyName(parsed.value)) parsed.value else null;
}

fn firstToolArgumentName(parameters_json: []const u8) ?[]const u8 {
    return firstStringInArrayAfterField(parameters_json, "\"required\"") orelse
        firstObjectKeyAfterField(parameters_json, "\"properties\"");
}

fn forcedSingleToolFirstArgumentName(tools: []const tool_format.ToolDefinition, tool_choice: ToolChoice) ?[]const u8 {
    if (tool_choice != .required or tools.len != 1) return null;
    return firstToolArgumentName(tools[0].parameters_json);
}

const ChatMessage = struct {
    role: []const u8 = "",
    content: Content = .{},
    tool_calls: []const HistoricalToolCall = &.{},
    tool_call_id: []const u8 = "",
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
    tools: []const RequestTool = &.{},
    tool_choice: ?std.json.Value = null,
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
    tools: []const tool_format.ToolDefinition,
    tool_choice: ToolChoice,
    injected_default_system: bool = false,
    /// Owns parameters_json strings and tool_calls rendering.
    arena: std.heap.ArenaAllocator,

    fn deinit(self: *ParsedChatRequest) void {
        self.parsed.deinit();
        self.arena.deinit();
        self.* = undefined;
    }
};

fn countValidChatMessages(messages: []const ChatMessage) usize {
    var count: usize = 0;
    for (messages) |message| {
        if (message.role.len == 0) continue;
        if (message.content.text.len == 0 and message.tool_calls.len == 0) continue;
        count += 1;
    }
    return count;
}

fn estimateChatPromptBytes(roles: []const []const u8, contents: []const []const u8, thinking_enabled: bool, tools: []const tool_format.ToolDefinition) usize {
    var total: usize = 128;
    const n = @min(roles.len, contents.len);
    for (0..n) |i| {
        total += roles[i].len + contents[i].len + 32;
    }
    if (thinking_enabled) total += thinking_prefix.len + 32 else total += 32;
    // Add budget for injected tool definitions: ~400 byte Qwen3 header + ~256 bytes per tool.
    if (tools.len > 0) {
        total += 512;
        for (tools) |t| {
            total += t.name.len + t.description.len + t.parameters_json.len + 64;
        }
        total += 192;
    }
    return total;
}

fn buildChatPrompt(allocator: std.mem.Allocator, tokenizer: *const tokenizer_mod.Tokenizer, roles: []const []const u8, contents: []const []const u8, enable_thinking: ?bool, skip_thinking_template: bool, tools: []const tool_format.ToolDefinition, tool_choice: ToolChoice, buf: []u8) ![]const u8 {
    const tf = tool_format.forTemplate(tokenizer.detectTemplateKind());
    const tools_for_template = if (tool_choice == .none) @as([]const tool_format.ToolDefinition, &.{}) else tools;
    const forced_tool_name = forcedSingleToolName(tools_for_template, tool_choice);
    const forced_tool_first_arg_name = forcedSingleToolFirstArgumentName(tools_for_template, tool_choice);
    return tokenizer.applyChatTemplateWithOptions(roles, contents, .{
        .enable_thinking = enable_thinking,
        .skip_thinking_template = skip_thinking_template,
        .tools = tools_for_template,
        .require_tool_call = tool_choice == .required,
        .forced_tool_call_name = forced_tool_name,
        .forced_tool_call_first_arg_name = forced_tool_first_arg_name,
        .tool_format = tf,
        .tool_render_allocator = allocator,
    }, buf);
}

fn allocChatPrompt(
    allocator: std.mem.Allocator,
    tokenizer: *const tokenizer_mod.Tokenizer,
    roles: []const []const u8,
    contents: []const []const u8,
    enable_thinking: ?bool,
    skip_thinking_template: bool,
    tools: []const tool_format.ToolDefinition,
    tool_choice: ToolChoice,
) ![]u8 {
    var capacity = estimateChatPromptBytes(roles, contents, supportsEnabledThinking(tokenizer, enable_thinking), tools);
    var attempts: u8 = 0;
    while (attempts < 4) : (attempts += 1) {
        const prompt_buf = try allocator.alloc(u8, capacity);
        errdefer allocator.free(prompt_buf);

        const prompt = buildChatPrompt(allocator, tokenizer, roles, contents, enable_thinking, skip_thinking_template, tools, tool_choice, prompt_buf) catch |err| {
            allocator.free(prompt_buf);
            if (err == error.BufferTooSmall) {
                capacity *= 2;
                continue;
            }
            return err;
        };

        return prompt_buf[0..prompt.len];
    }

    return error.BufferTooSmall;
}

fn buildChatTranscriptPrompt(
    allocator: std.mem.Allocator,
    tokenizer: *const tokenizer_mod.Tokenizer,
    roles: []const []const u8,
    contents: []const []const u8,
    enable_thinking: ?bool,
    skip_thinking_template: bool,
    tools: []const tool_format.ToolDefinition,
    tool_choice: ToolChoice,
    buf: []u8,
) ![]const u8 {
    const tf = tool_format.forTemplate(tokenizer.detectTemplateKind());
    const tools_for_template = if (tool_choice == .none) @as([]const tool_format.ToolDefinition, &.{}) else tools;
    return tokenizer.applyChatTemplateWithOptions(roles, contents, .{
        .add_generation_prompt = false,
        .enable_thinking = enable_thinking,
        .skip_thinking_template = skip_thinking_template,
        .tools = tools_for_template,
        .require_tool_call = tool_choice == .required,
        .tool_format = tf,
        .tool_render_allocator = allocator,
    }, buf);
}

fn isTransientChatReuseHint(role: []const u8, content: []const u8) bool {
    if (!std.mem.eql(u8, role, "system")) return false;
    const trimmed = std.mem.trimLeft(u8, content, " \t\r\n");
    return std.mem.startsWith(u8, trimmed, "Tool path guard:") or
        std.mem.startsWith(u8, trimmed, "OpenCode continuation guard:");
}

fn persistentChatHistoryLen(roles: []const []const u8, contents: []const []const u8) usize {
    var len = @min(roles.len, contents.len);
    while (len > 0 and isTransientChatReuseHint(roles[len - 1], contents[len - 1])) {
        len -= 1;
    }
    return len;
}

fn shouldForceDisableThinking(managed_id: ?[]const u8, model_path: []const u8, display_name: []const u8) bool {
    if (comptime !runtime.supports_model_management) return false;
    const entry = catalog_mod.findForLoadedModel(managed_id, model_path, display_name) orelse return false;
    return !entry.thinking_stable;
}

fn shouldSkipThinkingTemplateForRequest(tools_len: usize, tool_choice: ToolChoice) bool {
    return tools_len > 0 and tool_choice != .none;
}

fn warmChatReuseCache(
    server_state: *ServerState,
    resources: *const model_manager_mod.LoadedResources,
    tokenizer: *const tokenizer_mod.Tokenizer,
    engine: *forward_mod.InferenceEngine,
    thinking_enabled: bool,
    skip_thinking_template: bool,
    tools: []const tool_format.ToolDefinition,
    tool_choice: ToolChoice,
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
    const persistent_count = persistentChatHistoryLen(roles, contents);
    const trimmed_transient_hints = persistent_count < @min(roles.len, contents.len);
    const transcript_count = persistent_count + 1;
    const transcript_roles = try allocator.alloc([]const u8, transcript_count);
    defer allocator.free(transcript_roles);
    const transcript_contents = try allocator.alloc([]const u8, transcript_count);
    defer allocator.free(transcript_contents);

    @memcpy(transcript_roles[0..persistent_count], roles[0..persistent_count]);
    @memcpy(transcript_contents[0..persistent_count], contents[0..persistent_count]);
    transcript_roles[persistent_count] = "assistant";
    transcript_contents[persistent_count] = assistant_content;

    const transcript_capacity = estimateChatPromptBytes(transcript_roles, transcript_contents, thinking_enabled, tools) + assistant_content.len + 512;
    const transcript_buf = try allocator.alloc(u8, transcript_capacity);
    defer allocator.free(transcript_buf);
    const transcript_prompt = try buildChatTranscriptPrompt(allocator, tokenizer, transcript_roles, transcript_contents, thinking_enabled, skip_thinking_template, tools, tool_choice, transcript_buf);
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
        !trimmed_transient_hints and
        transcript_tokens.len >= processed_prefix_len and
        prompt_mismatch == null and
        response_mismatch == null;

    if (can_incremental) {
        const suffix_tokens = transcript_tokens[processed_prefix_len..];
        if (suffix_tokens.len > 0) {
            try engine.prefillBatched(state, suffix_tokens);
        }
        log.info("chat cache updated: session={s} prefix={d} suffix={d}", .{
            session_id,
            transcript_tokens.len,
            transcript_tokens.len - processed_prefix_len,
        });
    } else {
        var rebuild_reason: []const u8 = if (trimmed_transient_hints) "transient_hints_trimmed" else "prefix_mismatch";
        if (state.position != processed_prefix_len) {
            rebuild_reason = "state_position_mismatch";
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
        if (transcript_tokens.len > resources.context_capacity_tokens) return error.ContextLengthExceeded;
        state.position = 0;
        state.generated_tokens.clearRetainingCapacity();
        try engine.prefillBatched(state, transcript_tokens);
        server_state.setActiveContextTokens(state.position);
        log.info("chat cache rebuilt canonical transcript: session={s} reason={s} prefix={d}", .{
            session_id,
            rebuild_reason,
            transcript_tokens.len,
        });
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

fn appendAssistantToolCallHistoryBlock(
    buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    name: []const u8,
    arguments_json: []const u8,
) !void {
    try buf.appendSlice(allocator, "<tool_call>\n{\"name\": \"");
    try buf.appendSlice(allocator, name);
    try buf.appendSlice(allocator, "\", \"arguments\": ");
    try buf.appendSlice(allocator, arguments_json);
    try buf.appendSlice(allocator, "}\n</tool_call>\n");
}

fn parseChatRequest(allocator: std.mem.Allocator, body: []const u8) !ParsedChatRequest {
    var parsed = try std.json.parseFromSlice(ChatRequestBody, allocator, body, .{
        .ignore_unknown_fields = true,
    });
    errdefer parsed.deinit();

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_alloc = arena.allocator();

    const tools_enabled = toolCallingEnabled();

    const json_arena_allocator = parsed.arena.allocator();
    const messages = parsed.value.messages;
    const roles = try json_arena_allocator.alloc([]const u8, messages.len + 1);
    const contents = try json_arena_allocator.alloc([]const u8, messages.len + 1);

    var count: usize = 0;
    var has_guiding_message = false;
    for (messages) |message| {
        if (message.role.len == 0) continue;
        const has_tool_calls = tools_enabled and message.tool_calls.len > 0;
        if (message.content.text.len == 0 and !has_tool_calls) continue;
        if (std.mem.eql(u8, message.role, "system") or std.mem.eql(u8, message.role, "developer")) {
            has_guiding_message = true;
        }
    }

    // The default "Answer directly. Never output self-referential planning"
    // system prompt fights a thinking turn — the <think> block is exactly
    // meta-planning, and Qwen loops forever trying to satisfy both sides.
    // Skip it when thinking is enabled and the caller hasn't supplied their own.
    const thinking_on = parsed.value.enable_thinking orelse false;
    var injected_default_system = false;
    if (!has_guiding_message and !thinking_on) {
        roles[count] = "system";
        contents[count] = default_chat_system_prompt;
        count += 1;
        injected_default_system = true;
    }

    for (messages) |message| {
        if (message.role.len == 0) continue;
        // Assistant messages with historical tool_calls must be replayed as
        // assistant-visible <tool_call> blocks so the next tool result/user turn
        // has the same semantic context the model originally produced.
        if (tools_enabled and std.mem.eql(u8, message.role, "assistant") and message.tool_calls.len > 0) {
            var rendered: std.ArrayList(u8) = .{};
            if (message.content.text.len > 0) {
                const sanitized = sanitizeAssistantHistoryContent(message.content.text);
                try rendered.appendSlice(arena_alloc, sanitized);
                if (sanitized.len > 0 and sanitized[sanitized.len - 1] != '\n') {
                    try rendered.appendSlice(arena_alloc, "\n");
                }
            }
            for (message.tool_calls) |tc| {
                try appendAssistantToolCallHistoryBlock(&rendered, arena_alloc, tc.function.name, tc.function.arguments);
            }
            roles[count] = "assistant";
            contents[count] = try rendered.toOwnedSlice(arena_alloc);
            count += 1;
            continue;
        }
        if (message.content.text.len == 0) continue;
        roles[count] = message.role;
        contents[count] = if (std.mem.eql(u8, message.role, "assistant"))
            sanitizeAssistantHistoryContent(message.content.text)
        else
            message.content.text;
        count += 1;
    }

    // Convert RequestTool → ToolDefinition. When tool calling is disabled,
    // pretend the request had no tools — downstream code (prompt builder,
    // response emitter, streaming detector) sees an empty slice and runs
    // the existing chat path unchanged.
    const tools_in: []const RequestTool = if (tools_enabled) parsed.value.tools else &.{};
    const tools_out = try arena_alloc.alloc(tool_format.ToolDefinition, tools_in.len);
    for (tools_in, 0..) |t, ti| {
        const params_json = try std.json.Stringify.valueAlloc(arena_alloc, t.function.parameters, .{});
        tools_out[ti] = .{
            .name = t.function.name,
            .description = t.function.description,
            .parameters_json = params_json,
        };
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
        .tools = tools_out,
        .tool_choice = if (tools_enabled) parseToolChoice(parsed.value.tool_choice) else .auto,
        .injected_default_system = injected_default_system,
        .arena = arena,
    };
}

fn dropInjectedDefaultSystemForGemma(parsed: *ParsedChatRequest, is_gemma: bool) void {
    if (!is_gemma or !parsed.injected_default_system) return;
    if (parsed.roles.len == 0 or parsed.contents.len == 0) return;
    if (!std.mem.eql(u8, parsed.roles[0], "system")) return;
    if (!std.mem.eql(u8, parsed.contents[0], default_chat_system_prompt)) return;

    // Gemma chat templates already provide their own model turn scaffold.
    // Keep user-supplied guidance, but do not add an invisible extra turn.
    parsed.roles = parsed.roles[1..];
    parsed.contents = parsed.contents[1..];
    parsed.injected_default_system = false;
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

fn extractHarmonyMessage(text: []const u8, prefix: []const u8) []const u8 {
    const start = std.mem.indexOf(u8, text, prefix) orelse return "";
    const body = text[start + prefix.len ..];
    const end = findFirstStop(body, harmony_stop_strs[0..]) orelse body.len;
    return std.mem.trim(u8, body[0..end], " \t\r\n");
}

fn normalizeStructuredAssistantOutput(
    tokenizer: *const tokenizer_mod.Tokenizer,
    text: []const u8,
    thinking_enabled: bool,
    buf: []u8,
) ![]const u8 {
    if (!std.mem.eql(u8, tokenizer.detectTemplateKindName(), "openai_moe")) return text;

    const analysis = extractHarmonyMessage(text, harmony_analysis_prefix);
    const final = extractHarmonyMessage(text, harmony_final_prefix);
    if (analysis.len == 0 and final.len == 0) return text;
    if (!thinking_enabled or analysis.len == 0) return final;

    const joiner = if (final.len > 0) "\n" else "";
    const close = "\n</think>";
    const total_len = thinking_prefix.len + analysis.len + close.len + joiner.len + final.len;
    if (total_len > buf.len) return error.BufferTooSmall;

    @memcpy(buf[0..thinking_prefix.len], thinking_prefix);
    var pos = thinking_prefix.len;
    @memcpy(buf[pos .. pos + analysis.len], analysis);
    pos += analysis.len;
    @memcpy(buf[pos .. pos + close.len], close);
    pos += close.len;
    if (joiner.len > 0) {
        @memcpy(buf[pos .. pos + joiner.len], joiner);
        pos += joiner.len;
    }
    if (final.len > 0) {
        @memcpy(buf[pos .. pos + final.len], final);
        pos += final.len;
    }
    return buf[0..pos];
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

fn deinitParsedChatmlAssistantOutput(parsed: *const tool_format.ParsedAssistantOutput, allocator: std.mem.Allocator) void {
    for (parsed.tool_calls) |c| {
        allocator.free(c.id);
        allocator.free(c.name);
        allocator.free(c.arguments_json);
    }
    allocator.free(parsed.tool_calls);
    allocator.free(parsed.text_content);
}

fn historyAssistantContentWithTools(
    tokenizer: *const tokenizer_mod.Tokenizer,
    tool_fmt: tool_format.ToolFormat,
    tools_active: bool,
    text: []const u8,
    transport_buf: []u8,
    tool_history_buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
) ![]const u8 {
    if (!tools_active or !std.mem.eql(u8, tokenizer.detectTemplateKindName(), "chatml")) {
        return historyAssistantContent(tokenizer, text, transport_buf);
    }

    const parsed_output = try tool_fmt.parseAssistantToolCalls(text, allocator);
    defer deinitParsedChatmlAssistantOutput(&parsed_output, allocator);
    if (parsed_output.tool_calls.len == 0) {
        return historyAssistantContent(tokenizer, text, transport_buf);
    }

    tool_history_buf.clearRetainingCapacity();
    const answer = assistantAnswerForHistory(parsed_output.text_content);
    if (answer.len > 0) {
        try tool_history_buf.appendSlice(allocator, answer);
        if (answer[answer.len - 1] != '\n') try tool_history_buf.appendSlice(allocator, "\n");
    }
    for (parsed_output.tool_calls) |tc| {
        try appendAssistantToolCallHistoryBlock(tool_history_buf, allocator, tc.name, tc.arguments_json);
    }

    return transportAssistantContent(tokenizer, tool_history_buf.items, false, transport_buf);
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

fn stripThinkingForDisabledResponse(text: []const u8, buf: []u8) ![]const u8 {
    var cursor: usize = 0;
    var out_len: usize = 0;

    while (cursor < text.len) {
        const next_open = std.mem.indexOfPos(u8, text, cursor, thinking_open_tag);
        const next_close = std.mem.indexOfPos(u8, text, cursor, thinking_close_tag);
        const cut_at = switch (next_open != null or next_close != null) {
            false => null,
            true => blk: {
                if (next_open == null) break :blk next_close;
                if (next_close == null) break :blk next_open;
                break :blk @min(next_open.?, next_close.?);
            },
        };

        if (cut_at == null) {
            const tail = text[cursor..];
            if (out_len + tail.len > buf.len) return error.BufferTooSmall;
            @memcpy(buf[out_len .. out_len + tail.len], tail);
            out_len += tail.len;
            break;
        }

        const cut = cut_at.?;
        if (cut > cursor) {
            const chunk = text[cursor..cut];
            if (out_len + chunk.len > buf.len) return error.BufferTooSmall;
            @memcpy(buf[out_len .. out_len + chunk.len], chunk);
            out_len += chunk.len;
        }

        if (next_open != null and next_open.? == cut) {
            const after_open = cut + thinking_open_tag.len;
            if (std.mem.indexOfPos(u8, text, after_open, thinking_close_tag)) |close| {
                cursor = close + thinking_close_tag.len;
                while (cursor < text.len and std.ascii.isWhitespace(text[cursor])) : (cursor += 1) {}
                continue;
            }
            if (out_len == 0) {
                const fallback = sanitizeAnswerTail(text[after_open..]);
                if (fallback.len > buf.len) return error.BufferTooSmall;
                @memcpy(buf[0..fallback.len], fallback);
                out_len = fallback.len;
            }
            break;
        }

        cursor = cut + thinking_close_tag.len;
        while (cursor < text.len and std.ascii.isWhitespace(text[cursor])) : (cursor += 1) {}
    }

    const stripped = std.mem.trim(u8, buf[0..out_len], " \t\r\n");
    if (stripped.len > 0) return stripped;

    const trimmed = std.mem.trimLeft(u8, text, " \t\r\n");
    if (!std.mem.startsWith(u8, trimmed, thinking_open_tag)) return stripped;

    const after_open = thinking_open_tag.len;
    const close = std.mem.indexOfPos(u8, trimmed, after_open, thinking_close_tag) orelse return stripped;
    const answer = std.mem.trim(u8, sanitizeAnswerTail(trimmed[close + thinking_close_tag.len ..]), " \t\r\n");
    if (answer.len > 0) return stripped;

    const reasoning = std.mem.trim(u8, sanitizeAnswerTail(trimmed[after_open..close]), " \t\r\n");
    if (reasoning.len == 0) return stripped;
    if (reasoning.len > buf.len) return error.BufferTooSmall;
    @memcpy(buf[0..reasoning.len], reasoning);
    return buf[0..reasoning.len];
}

fn hasDanglingTrailingQuote(text: []const u8) bool {
    const trimmed = std.mem.trimRight(u8, text, " \t\r\n");
    if (trimmed.len == 0 or trimmed[trimmed.len - 1] != '"') return false;
    var body = trimmed[0 .. trimmed.len - 1];
    while (std.mem.endsWith(u8, body, utf8_replacement)) {
        body = body[0 .. body.len - utf8_replacement.len];
    }
    body = std.mem.trimRight(u8, body, " \t\r\n");
    if (body.len == 0) return true;
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

// Returns the byte length of the longest prefix of `bytes` that ends on a
// complete UTF-8 codepoint boundary. Trailing bytes that form an incomplete
// sequence (e.g. one or two bytes of a 4-byte emoji) are excluded so they can
// be carried into the next streaming chunk. Byte-level BPE tokenizers (Qwen,
// GPT-2) routinely split multi-byte characters across single-byte tokens; if
// those bytes are shipped to the browser one event at a time, TextDecoder
// emits U+FFFD per orphan byte.
fn lastCompleteUtf8End(bytes: []const u8) usize {
    if (bytes.len == 0) return 0;
    // Walk back over up to 3 trailing continuation bytes (10xxxxxx).
    var i: usize = bytes.len;
    var continuations: usize = 0;
    while (i > 0 and continuations < 3) {
        const b = bytes[i - 1];
        if ((b & 0xC0) != 0x80) break;
        i -= 1;
        continuations += 1;
    }
    if (i == 0) return bytes.len; // all continuations, nothing to anchor on — flush as-is
    const lead = bytes[i - 1];
    const expected: usize = if (lead < 0x80) 1 else if ((lead & 0xE0) == 0xC0) 2 else if ((lead & 0xF0) == 0xE0) 3 else if ((lead & 0xF8) == 0xF0) 4 else 1; // malformed lead — let it through
    const have = bytes.len - (i - 1);
    if (have >= expected) return bytes.len;
    return i - 1;
}

fn pendingGeneratedText(text: []const u8, sent_len: *usize, hold_back: usize) []const u8 {
    const emit_end = if (text.len > hold_back) text.len - hold_back else 0;
    if (sent_len.* > emit_end) sent_len.* = emit_end;
    if (emit_end <= sent_len.*) return "";
    const pending = text[sent_len.*..emit_end];
    sent_len.* = emit_end;
    return pending;
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

/// Validates that the model the client requested is the one currently loaded;
/// if not, attempts to swap to it. Returns `true` if generation may proceed,
/// `false` if an error response was already sent.
///
/// Caller must already hold the shared generation lock (i.e. have an active
/// `GenerationGuard`), since `activateManagedModel` requires it.
fn ensureRequestedModelActive(
    conn: *http.Connection,
    manager: *model_manager_mod.ModelManager,
    server_state: *ServerState,
    requested_model: []const u8,
) !bool {
    if (requested_model.len == 0) return true;

    if (manager.currentResources()) |current| {
        if (current.managed_id) |active_id| {
            if (std.mem.eql(u8, active_id, requested_model)) return true;
        }
        if (comptime runtime.supports_model_management) {
            // The model may have been loaded by --model <path> with no managed_id;
            // resolve the catalog entry that corresponds to the on-disk file so
            // a request for that catalog id is treated as a no-op rather than a swap.
            if (catalog_mod.findForLoadedModel(current.managed_id, current.model_path, current.display_name)) |entry| {
                if (std.mem.eql(u8, entry.id, requested_model)) return true;
            }
        }
    }

    if (comptime !runtime.supports_model_management) {
        try conn.sendError(404, "model_not_found", "Requested model is not loaded; this build does not support runtime model swapping");
        return false;
    }

    manager.activateManagedModel(requested_model, true, false) catch |err| {
        const status: u16 = switch (err) {
            error.UnknownManagedModel, error.ModelNotInstalled => 404,
            error.GpuAlreadyReserved => 409,
            else => 400,
        };
        const code = switch (err) {
            error.UnknownManagedModel, error.ModelNotInstalled => "model_not_found",
            else => "invalid_request_error",
        };
        const msg = switch (err) {
            error.UnknownManagedModel => "Unknown model id",
            error.ModelNotInstalled => "Model is not installed in the local cache",
            error.ModelUnsupportedOnThisGpu => "Model is not validated on the current GPU profile. Activate it explicitly with \"force\":true first.",
            error.ModelDoesNotFit => "Model does not fit the current GPU memory budget",
            error.GpuAlreadyReserved => "Another zinc process owns this GPU",
            else => @errorName(err),
        };
        log.warn("Model activation failed for '{s}': {s}", .{ requested_model, @errorName(err) });
        try conn.sendError(status, code, msg);
        return false;
    };
    server_state.clearChatReuseCache();
    return true;
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
    if (!try ensureRequestedModelActive(conn, manager, server_state, parsed.parsed.value.model)) return;
    const resources = manager.currentResources() orelse {
        try conn.sendError(503, "service_unavailable", "No model is currently loaded");
        return;
    };
    const engine = &resources.engine;
    const tokenizer = &resources.tokenizer;
    const model_name = resources.display_name;
    dropInjectedDefaultSystemForGemma(&parsed, resources.model.config.architecture == .gemma);

    // If the catalog marks this model's thinking as unstable, force-disable
    // visible thinking but still keep the tokenizer's no-thinking generation
    // suffix. Some Qwen thinking models stop immediately after a bare assistant
    // header, but answer correctly after the closed <think></think> scaffold.
    const force_disable_thinking = shouldForceDisableThinking(resources.managed_id, resources.model_path, resources.display_name);
    if (force_disable_thinking) {
        parsed.enable_thinking = false;
    }
    // Tool-calling requests skip the Qwen empty-thinking scaffold so the
    // assistant can begin directly with a <tool_call> block.
    const skip_thinking_template = shouldSkipThinkingTemplateForRequest(parsed.tools.len, parsed.tool_choice);
    if (skip_thinking_template) {
        parsed.enable_thinking = null;
    }
    if (comptime !runtime.supports_sampling_controls) {
        if (parsed.temperature > 0.0001 or parsed.top_p < 0.9999) {
            try conn.sendError(400, "invalid_request_error", "Metal server currently supports greedy decoding only (temperature=0, top_p=1)");
            return;
        }
    }
    const sampling = runtime.SamplingParams{
        .temperature = if (parsed.temperature <= 0.0001) 0.0 else std.math.clamp(parsed.temperature, 0.0, 2.0),
        .top_p = std.math.clamp(parsed.top_p, 0.0, 1.0),
        .repetition_penalty = if (parsed.temperature > 0.0001 or parsed.top_p < 0.9999) 1.08 else 1.0,
        .top_k = 64,
    };
    const previous_logits_readback = runtime.logitsReadbackEnabled(engine);
    if (sampling.requiresLogitsReadback() and !previous_logits_readback) {
        runtime.enableLogitsReadback(engine);
    }
    defer runtime.setLogitsReadbackEnabled(engine, previous_logits_readback);

    const prompt = allocChatPrompt(allocator, tokenizer, parsed.roles, parsed.contents, parsed.enable_thinking, skip_thinking_template, parsed.tools, parsed.tool_choice) catch |err| {
        if (err == error.OutOfMemory) {
            try conn.sendError(500, "internal_error", "Prompt allocation failed");
            return;
        }
        if (err == error.BufferTooSmall) {
            try conn.sendError(400, "invalid_request_error", "Prompt too long");
            return;
        }
        try conn.sendError(500, "internal_error", "Prompt formatting failed");
        return;
    };
    defer allocator.free(prompt);

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
    const prompt_token_count: u32 = @intCast(@min(prompt_tokens.len, std.math.maxInt(u32)));
    if (prompt_token_count > resources.context_capacity_tokens) {
        try conn.sendError(400, "context_length_exceeded", "Prompt exceeds context capacity");
        return;
    }
    server_state.setActiveContextTokens(prompt_token_count);

    const ts = @divTrunc(std.time.timestamp(), 1);
    const request_budget = memory_plan.requestBudget(
        prompt_token_count,
        parsed.max_tokens,
        resources.context_capacity_tokens,
    );
    const max_tokens = request_budget.completion_tokens;
    const seed_ns: i128 = std.time.nanoTimestamp();
    var req_id_buf: [32]u8 = undefined;
    const req_id = std.fmt.bufPrint(&req_id_buf, "chatcmpl-{x}", .{@as(u64, @truncate(@as(u128, @bitCast(seed_ns))))}) catch "chatcmpl-0";
    const thinking_enabled = supportsEnabledThinking(tokenizer, parsed.enable_thinking);
    const tools_for_choice = if (parsed.tool_choice == .none) @as([]const tool_format.ToolDefinition, &.{}) else parsed.tools;
    const forced_tool_name = forcedSingleToolName(tools_for_choice, parsed.tool_choice);
    const forced_tool_first_arg_name = forcedSingleToolFirstArgumentName(tools_for_choice, parsed.tool_choice);
    const seed_bits: u128 = @bitCast(seed_ns);
    var prng = std.Random.DefaultPrng.init(@truncate(seed_bits));
    const random = prng.random();
    const cacheable_session = parsed.session_id.len > 0;

    var state = forward_mod.DecodeState.init(allocator);
    defer state.deinit();
    state.requested_context_tokens = request_budget.target_context_tokens;
    if (max_tokens < parsed.max_tokens) {
        log.info("Clamped chat decode budget from {d} to {d} tokens (prompt={d}, capacity={d})", .{
            parsed.max_tokens,
            max_tokens,
            prompt_token_count,
            resources.context_capacity_tokens,
        });
    }

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
                    skip_thinking_template,
                    parsed.tools,
                    parsed.tool_choice,
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
        if (thinking_enabled and forced_tool_name == null) {
            streamText(conn, thinking_prefix, req_id, ts, model_name) catch return;
        }
        if (conn.isPeerClosed()) return;
    }

    const reused_prefix_len = if (cacheable_session)
        server_state.chat_reuse_cache.matchingPrefixLen(parsed.session_id, resources.model_path, prompt_tokens, std.time.nanoTimestamp())
    else
        0;
    const prefill_start_ns = std.time.nanoTimestamp();
    const prefill_work_tokens = if (reused_prefix_len > 0)
        prompt_tokens.len - reused_prefix_len
    else
        prompt_tokens.len;
    if (reused_prefix_len > 0) {
        state.position = @intCast(reused_prefix_len);
        if (reused_prefix_len < prompt_tokens.len) {
            engine.prefillBatched(&state, prompt_tokens[reused_prefix_len..]) catch |err| {
                log.err("Chat prefill failed after cache hit: {s}", .{@errorName(err)});
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
        engine.prefillBatched(&state, prompt_tokens) catch |err| {
            log.err("Chat prefill failed: {s}", .{@errorName(err)});
            if (parsed.stream) {
                conn.writeSseDone() catch {};
            } else {
                try conn.sendError(500, "internal_error", "Prefill failed");
            }
            return;
        };
    }
    const prefill_end_ns = std.time.nanoTimestamp();
    logPrefillTiming(prefill_work_tokens, prefill_start_ns, prefill_end_ns);
    server_state.setActiveContextTokens(state.position);

    if (parsed.stream) {
        // Decode loop with buffered stop detection.
        // Tokens are buffered and only sent once we confirm they're not part of <|im_end|>.
        // Tokenizer decides which IDs terminate a turn (e.g. Gemma 4 treats <eos>=1
        // and </s>=212 as EOG alongside <turn|>=106, while Qwen's token 1 is a
        // normal `"` character).
        const isEog = struct {
            fn check(tok: *const tokenizer_mod.Tokenizer, token: u32) bool {
                return tok.isEndOfGeneration(token);
            }
        }.check;
        const stop_strs = chat_stop_strs[0..];
        var gen_text_buf: [32768]u8 = undefined; // accumulated decoded text for stop check
        var gen_text_len: usize = 0;
        var sent_text_len: usize = 0; // how much of gen_text has been confirmed safe to send
        var sent_visible_len: usize = 0; // cleaned visible bytes already streamed when thinking is disabled
        var visible_buf: [4096]u8 = undefined;
        var stopped = false;
        var finish_reason: FinishReason = if (max_tokens == 0 and parsed.max_tokens > 0) .length else .stop;

        // Streaming tool-call detector. The detector is allocated unconditionally
        // (so deinit always works) but `tools_active` short-circuits the per-chunk
        // detection logic when the request didn't ask for tools — in that case
        // bytes flow through directly via streamText, matching mainline latency.
        const tools_active = parsed.tools.len > 0;
        const stream_tool_fmt = tool_format.forTemplate(tokenizer.detectTemplateKind());
        const stream_detector = try stream_tool_fmt.newStreamingDetector(allocator);
        defer {
            stream_detector.deinit();
            allocator.destroy(stream_detector);
        }
        var any_tool_call_emitted = false;
        var tool_call_index: u32 = 0;
        var generated: u32 = 0;
        const decode_start_ns = std.time.nanoTimestamp();

        if (max_tokens > 0) {
            var prev_token = runtime.sample(engine, &state, sampling, random);
            state.generated_tokens.append(allocator, prev_token) catch {};

            while (generated < max_tokens and !isEog(tokenizer, prev_token) and !stopped) {
                if (conn.isPeerClosed()) return;

                // Accumulate this token's decoded text
                var dec_buf: [256]u8 = undefined;
                const tok_text = tokenizer.decodeToken(prev_token, &dec_buf);
                if (isReplacementArtifact(tok_text)) {
                    if (generated < max_tokens) {
                        if (conn.isPeerClosed()) return;
                        runtime.decodeStep(engine, &state, prev_token, true) catch break;
                        processed_generated_tokens.append(allocator, prev_token) catch {};
                        server_state.setActiveContextTokens(state.position);
                        if (conn.isPeerClosed()) return;
                        prev_token = runtime.sample(engine, &state, sampling, random);
                        state.generated_tokens.append(allocator, prev_token) catch {};
                        generated += 1;
                        continue;
                    }
                    break;
                }
                if (gen_text_len + tok_text.len < gen_text_buf.len) {
                    @memcpy(gen_text_buf[gen_text_len..][0..tok_text.len], tok_text);
                    gen_text_len += tok_text.len;
                }

                // Translate Gemma 4 thinking channel to <think> tags for the chat UI.
                // Model generates: <|channel>thought\n...<channel|>response
                // We convert to: <think>\n...\n</think>\nresponse
                if (sent_text_len == 0 and gen_text_len >= 18) {
                    const gemma_open = "<|channel>thought\n";
                    const gemma_empty = "<|channel>thought\n<channel|>";
                    if (std.mem.startsWith(u8, gen_text_buf[0..gen_text_len], gemma_empty)) {
                        // Empty thinking: replace with <think>\n</think>\n
                        const replacement = "<think>\n</think>\n";
                        @memcpy(gen_text_buf[0..replacement.len], replacement);
                        if (gen_text_len > gemma_empty.len) {
                            const rest_len = gen_text_len - gemma_empty.len;
                            std.mem.copyForwards(u8, gen_text_buf[replacement.len..], gen_text_buf[gemma_empty.len..][0..rest_len]);
                        }
                        gen_text_len = gen_text_len - gemma_empty.len + replacement.len;
                    } else if (std.mem.startsWith(u8, gen_text_buf[0..gen_text_len], gemma_open)) {
                        // Has thinking content: replace <|channel>thought\n with <think>\n
                        const replacement = "<think>\n";
                        @memcpy(gen_text_buf[0..replacement.len], replacement);
                        if (gen_text_len > gemma_open.len) {
                            const rest_len = gen_text_len - gemma_open.len;
                            std.mem.copyForwards(u8, gen_text_buf[replacement.len..], gen_text_buf[gemma_open.len..][0..rest_len]);
                        }
                        gen_text_len = gen_text_len - gemma_open.len + replacement.len;
                    }
                }
                // Also replace <channel|> with </think>\n anywhere in the text
                {
                    const gemma_close = "<channel|>";
                    const think_close = "</think>\n";
                    if (gen_text_len >= gemma_close.len) {
                        if (std.mem.indexOf(u8, gen_text_buf[0..gen_text_len], gemma_close)) |idx| {
                            @memcpy(gen_text_buf[idx..][0..think_close.len], think_close);
                            if (gen_text_len > idx + gemma_close.len) {
                                const rest_start = idx + gemma_close.len;
                                const dest_start = idx + think_close.len;
                                const rest_len = gen_text_len - rest_start;
                                std.mem.copyForwards(u8, gen_text_buf[dest_start..], gen_text_buf[rest_start..][0..rest_len]);
                            }
                            gen_text_len = gen_text_len - gemma_close.len + think_close.len;
                        }
                    }
                }

                // Check for explicit chat stops, reopened think blocks, and leaked prompt-analysis tails.
                if (findStreamingStopStart(gen_text_buf[0..gen_text_len])) |stop_idx| {
                    gen_text_len = stop_idx;
                    const pending_text = gen_text_buf[sent_text_len..gen_text_len];
                    const cleaned_pending = trimTrailingChatArtifacts(pending_text);
                    gen_text_len = sent_text_len + cleaned_pending.len;
                    if (cleaned_pending.len > 0 and forced_tool_name == null) {
                        streamTextViaDetector(conn, cleaned_pending, req_id, ts, model_name, stream_detector, &any_tool_call_emitted, &tool_call_index, allocator, tools_active) catch return;
                    }
                    sent_text_len = gen_text_len;
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
                    if (forced_tool_name != null) {
                        sent_text_len = gen_text_len;
                    } else if (thinking_enabled) {
                        // Stream the accumulated bytes since the last safe send,
                        // trimmed to a UTF-8 codepoint boundary so partial multi-
                        // byte chars (emojis, CJK) carry into the next iteration.
                        const pending_slice = gen_text_buf[sent_text_len..gen_text_len];
                        const safe_end_rel = lastCompleteUtf8End(pending_slice);
                        if (safe_end_rel > 0) {
                            streamTextViaDetector(conn, pending_slice[0..safe_end_rel], req_id, ts, model_name, stream_detector, &any_tool_call_emitted, &tool_call_index, allocator, tools_active) catch return;
                        }
                        sent_text_len += safe_end_rel;
                    } else {
                        const visible = stripThinkingForDisabledResponse(gen_text_buf[0..gen_text_len], &visible_buf) catch gen_text_buf[0..gen_text_len];
                        const safe_visible = lastCompleteUtf8End(visible);
                        if (safe_visible > sent_visible_len) {
                            streamTextViaDetector(conn, visible[sent_visible_len..safe_visible], req_id, ts, model_name, stream_detector, &any_tool_call_emitted, &tool_call_index, allocator, tools_active) catch return;
                            sent_visible_len = safe_visible;
                        }
                        sent_text_len = gen_text_len;
                    }
                }

                generated += 1;

                // Generate next token
                if (generated < max_tokens) {
                    if (conn.isPeerClosed()) return;
                    runtime.decodeStep(engine, &state, prev_token, true) catch break;
                    processed_generated_tokens.append(allocator, prev_token) catch {};
                    server_state.setActiveContextTokens(state.position);
                    if (conn.isPeerClosed()) return;
                    prev_token = runtime.sample(engine, &state, sampling, random);
                    state.generated_tokens.append(allocator, prev_token) catch {};
                } else break;
            }

            if (!stopped and !isEog(tokenizer, prev_token) and generated >= max_tokens) {
                finish_reason = .length;
            }

            // Flush any remaining pending tokens (only if we didn't hit stop)
            if (!stopped) {
                if (thinking_enabled) {
                    const pending_text = gen_text_buf[sent_text_len..gen_text_len];
                    const cleaned_pending = trimTrailingChatArtifacts(pending_text);
                    if (cleaned_pending.len > 0 and forced_tool_name == null) {
                        streamTextViaDetector(conn, cleaned_pending, req_id, ts, model_name, stream_detector, &any_tool_call_emitted, &tool_call_index, allocator, tools_active) catch return;
                    }
                } else {
                    const visible = stripThinkingForDisabledResponse(gen_text_buf[0..gen_text_len], &visible_buf) catch gen_text_buf[0..gen_text_len];
                    const cleaned_visible = trimTrailingChatArtifacts(visible);
                    if (cleaned_visible.len > sent_visible_len and forced_tool_name == null) {
                        streamTextViaDetector(conn, cleaned_visible[sent_visible_len..], req_id, ts, model_name, stream_detector, &any_tool_call_emitted, &tool_call_index, allocator, tools_active) catch return;
                    }
                }
            }
            // Flush detector tail (bytes held while waiting for a potential tool call tag)
            if (forced_tool_name == null) {
                const tail = stream_detector.finalize();
                if (tail.len > 0) {
                    streamText(conn, tail, req_id, ts, model_name) catch return;
                }
            }
        }
        const decode_end_ns = std.time.nanoTimestamp();
        logGeneratedTiming(generated, decode_start_ns, decode_end_ns);

        if (forced_tool_name) |tool_name| {
            const args_json = try forcedToolArgumentsJsonAlloc(allocator, forced_tool_first_arg_name, gen_text_buf[0..gen_text_len]);
            defer allocator.free(args_json);
            var forced_id_buf: [64]u8 = undefined;
            const forced_id = std.fmt.bufPrint(&forced_id_buf, "call_forced_{x}", .{@as(u64, @truncate(@as(u128, @bitCast(seed_ns))))}) catch "call_forced";
            const ev = try formatStreamingToolCallChunk(allocator, req_id, ts, model_name, tool_call_index, forced_id, tool_name, args_json);
            defer allocator.free(ev);
            try conn.writeSseEvent(ev);
            any_tool_call_emitted = true;
            tool_call_index += 1;
        }

        // Final chunk with finish_reason
        if (any_tool_call_emitted and finish_reason == .stop) {
            finish_reason = .tool_calls;
        }
        {
            var chunk_buf: [1024]u8 = undefined;
            const chunk = std.fmt.bufPrint(&chunk_buf,
                \\{{"id":"{s}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{}},"finish_reason":"{s}"}}]}}
            , .{ req_id, ts, model_name, @tagName(finish_reason) }) catch "";
            conn.writeSseEvent(chunk) catch return;
        }

        conn.writeSseDone() catch return;
        if (cacheable_session and forced_tool_name == null) {
            const trimmed_stream_text = sanitizeAssistantHistoryContent(gen_text_buf[0..gen_text_len]);
            var transport_buf: [32768]u8 = undefined;
            var tool_history_buf: std.ArrayList(u8) = .{};
            defer tool_history_buf.deinit(allocator);
            const transport_text = historyAssistantContentWithTools(tokenizer, stream_tool_fmt, tools_active, trimmed_stream_text, &transport_buf, &tool_history_buf, allocator) catch trimmed_stream_text;
            cache_assistant_text = allocator.dupe(u8, transport_text) catch null;
        }
    } else {
        // Non-streaming: use same prefill+decode loop with stop detection
        var text_buf: std.ArrayList(u8) = .{};
        defer text_buf.deinit(allocator);
        var ns_gen: u32 = 0;
        const nsIsEog = struct {
            fn check(tok: *const tokenizer_mod.Tokenizer, token: u32) bool {
                return tok.isEndOfGeneration(token);
            }
        }.check;
        var finish_reason: FinishReason = if (max_tokens == 0 and parsed.max_tokens > 0) .length else .stop;
        const decode_start_ns = std.time.nanoTimestamp();
        if (max_tokens > 0) {
            var prev = runtime.sample(engine, &state, sampling, random);
            state.generated_tokens.append(allocator, prev) catch {};
            while (ns_gen < max_tokens and !nsIsEog(tokenizer, prev)) {
                var decode_buf2: [256]u8 = undefined;
                const tok_utf8 = tokenizer.decodeToken(prev, &decode_buf2);
                if (isReplacementArtifact(tok_utf8)) {
                    runtime.decodeStep(engine, &state, prev, true) catch break;
                    processed_generated_tokens.append(allocator, prev) catch {};
                    server_state.setActiveContextTokens(state.position);
                    prev = runtime.sample(engine, &state, sampling, random);
                    state.generated_tokens.append(allocator, prev) catch {};
                    continue;
                }
                text_buf.appendSlice(allocator, tok_utf8) catch break;
                ns_gen += 1;
                const hit = if (findStreamingStopStart(text_buf.items)) |pos| blk: {
                    text_buf.shrinkRetainingCapacity(pos);
                    break :blk true;
                } else false;
                if (hit) break;
                if (ns_gen >= max_tokens) break;
                runtime.decodeStep(engine, &state, prev, true) catch break;
                processed_generated_tokens.append(allocator, prev) catch {};
                server_state.setActiveContextTokens(state.position);
                prev = runtime.sample(engine, &state, sampling, random);
                state.generated_tokens.append(allocator, prev) catch {};
            }
            if (!nsIsEog(tokenizer, prev) and ns_gen >= max_tokens and findStreamingStopStart(text_buf.items) == null) {
                finish_reason = .length;
            }
        }
        const decode_end_ns = std.time.nanoTimestamp();
        logGeneratedTiming(ns_gen, decode_start_ns, decode_end_ns);

        // Escape the full text for JSON
        var structured_buf: [16384]u8 = undefined;
        const structured_text = normalizeStructuredAssistantOutput(tokenizer, text_buf.items, thinking_enabled, &structured_buf) catch text_buf.items;
        var strip_buf: [16384]u8 = undefined;
        const base_text = if (thinking_enabled)
            structured_text
        else
            stripThinkingForDisabledResponse(structured_text, &strip_buf) catch structured_text;
        const trimmed_text = trimTrailingChatArtifacts(base_text);
        var thinking_buf: [16384]u8 = undefined;
        const prefixed_text = prefixThinkingEnvelope(trimmed_text, thinking_enabled, &thinking_buf) catch trimmed_text;
        var sanitized_thinking_buf: [16384]u8 = undefined;
        const response_text = if (thinking_enabled)
            sanitizeThinkingOutput(prefixed_text, &sanitized_thinking_buf) catch prefixed_text
        else
            prefixed_text;

        // Parse tool calls from the response text — only when the request
        // actually carried tools (and tool calling is enabled). Skipping for
        // tool-less requests avoids spending a parser pass on every chat
        // completion and keeps the path identical to mainline pre-merge.
        const tool_fmt = tool_format.forTemplate(tokenizer.detectTemplateKind());
        const parsed_output = if (parsed.tools.len > 0)
            try tool_fmt.parseAssistantToolCalls(response_text, allocator)
        else
            tool_format.ParsedAssistantOutput{ .text_content = "", .tool_calls = &.{} };
        defer if (parsed.tools.len > 0) {
            for (parsed_output.tool_calls) |c| {
                allocator.free(c.id);
                allocator.free(c.name);
                allocator.free(c.arguments_json);
            }
            allocator.free(parsed_output.tool_calls);
            allocator.free(parsed_output.text_content);
        };

        if (parsed_output.tool_calls.len > 0 and finish_reason == .stop) {
            finish_reason = .tool_calls;
        }

        if (parsed_output.tool_calls.len > 0) {
            // Emit response with tool_calls, content null
            var wb: std.ArrayList(u8) = .{};
            defer wb.deinit(allocator);
            try wb.writer(allocator).print(
                \\{{"id":"{s}","object":"chat.completion","created":{d},"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":null,"tool_calls":[
            , .{ req_id, ts, model_name });
            for (parsed_output.tool_calls, 0..) |tc, ti| {
                if (ti > 0) try wb.appendSlice(allocator, ",");
                var args_esc_buf: [16384]u8 = undefined;
                const args_esc = jsonEscape(tc.arguments_json, &args_esc_buf);
                try wb.writer(allocator).print(
                    \\{{"id":"{s}","type":"function","function":{{"name":"{s}","arguments":"{s}"}}}}
                , .{ tc.id, tc.name, args_esc });
            }
            try wb.writer(allocator).print(
                \\]}},"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
            , .{ @tagName(finish_reason), prompt_tokens.len, ns_gen, prompt_tokens.len + ns_gen });
            try conn.sendJson(200, wb.items);
        } else {
            var escaped_buf: [16384]u8 = undefined;
            const escaped_text = jsonEscape(response_text, &escaped_buf);
            var resp_fixed_buf: [32768]u8 = undefined;
            const resp = std.fmt.bufPrint(&resp_fixed_buf,
                \\{{"id":"{s}","object":"chat.completion","created":{d},"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"}},"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
            , .{
                req_id,       ts,                         model_name,
                escaped_text, @tagName(finish_reason),    prompt_tokens.len,
                ns_gen,       prompt_tokens.len + ns_gen,
            }) catch {
                try conn.sendError(500, "internal_error", "Response too large");
                return;
            };
            try conn.sendJson(200, resp);
        }
        if (cacheable_session) {
            var transport_buf: [32768]u8 = undefined;
            var tool_history_buf: std.ArrayList(u8) = .{};
            defer tool_history_buf.deinit(allocator);
            const transport_text = historyAssistantContentWithTools(tokenizer, tool_fmt, parsed.tools.len > 0, response_text, &transport_buf, &tool_history_buf, allocator) catch response_text;
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
    if (!try ensureRequestedModelActive(conn, manager, server_state, parsed.model_id)) return;
    const resources = manager.currentResources() orelse {
        try conn.sendError(503, "service_unavailable", "No model is currently loaded");
        return;
    };
    const tokenizer = &resources.tokenizer;
    const engine = &resources.engine;
    const model_name = resources.display_name;

    // Tokenize raw prompt (no chat template)
    const prompt_tokens = tokenizer.encodePrompt(parsed.prompt, allocator) catch {
        try conn.sendError(500, "internal_error", "Tokenization failed");
        return;
    };
    defer allocator.free(prompt_tokens);
    defer server_state.clearActiveContext();
    if (prompt_tokens.len == 0) {
        try conn.sendError(500, "internal_error", "Tokenization produced no prompt tokens");
        return;
    }
    const prompt_token_count: u32 = @intCast(@min(prompt_tokens.len, std.math.maxInt(u32)));
    if (prompt_token_count > resources.context_capacity_tokens) {
        try conn.sendError(400, "context_length_exceeded", "Prompt exceeds context capacity");
        return;
    }
    server_state.setActiveContextTokens(prompt_token_count);
    const request_budget = memory_plan.requestBudget(
        prompt_token_count,
        parsed.max_tokens,
        resources.context_capacity_tokens,
    );
    const max_tokens = request_budget.completion_tokens;
    if (max_tokens < parsed.max_tokens) {
        log.info("Clamped completion decode budget from {d} to {d} tokens (prompt={d}, capacity={d})", .{
            parsed.max_tokens,
            max_tokens,
            prompt_token_count,
            resources.context_capacity_tokens,
        });
    }

    const ts = @divTrunc(std.time.timestamp(), 1);
    const seed_ns: i128 = std.time.nanoTimestamp();
    var req_id_buf: [32]u8 = undefined;
    const req_id = std.fmt.bufPrint(&req_id_buf, "cmpl-{x}", .{@as(u64, @truncate(@as(u128, @bitCast(seed_ns))))}) catch "cmpl-0";

    const output_tokens = forward_mod.generate(engine, prompt_tokens, max_tokens, tokenizer.eosId(), allocator) catch |err| {
        log.err("Completion generation failed: {s}", .{@errorName(err)});
        try conn.sendError(500, "internal_error", "Generation failed");
        return;
    };
    defer allocator.free(output_tokens);

    var text_buf: std.ArrayList(u8) = .{};
    defer text_buf.deinit(allocator);
    var decode_buf: [512]u8 = undefined;
    for (output_tokens) |tid| {
        // decodeToken reverses the GPT-2 byte-level remapping (e.g. `Ġ` → ' ').
        // Appending the raw vocab string directly leaked those marker characters
        // into /v1/completions responses.
        const t = tokenizer.decodeToken(tid, &decode_buf);
        text_buf.appendSlice(allocator, t) catch break;
    }

    const finish_reason = completionFinishReason(parsed.max_tokens, max_tokens, output_tokens.len);

    var escaped_buf: [16384]u8 = undefined;
    const escaped_text = jsonEscape(text_buf.items, &escaped_buf);

    var resp_buf: [32768]u8 = undefined;
    const resp = std.fmt.bufPrint(&resp_buf,
        \\{{"id":"{s}","object":"text_completion","created":{d},"model":"{s}","choices":[{{"index":0,"text":"{s}","finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
    , .{
        req_id,            ts,                                    model_name,
        escaped_text,      @tagName(finish_reason),               prompt_tokens.len,
        output_tokens.len, prompt_tokens.len + output_tokens.len,
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

/// Minimal JSON field extraction (no full parser — just find key fields).
const ParsedJsonFields = struct {
    model_id: []const u8,
    messages_content: []const u8, // last user message content
    prompt_text: []const u8, // raw prompt for /v1/completions
    max_tokens: u32,
    stream: bool,
    force: bool,
    temperature: f32,
    enable_thinking: ?bool,
};

fn parseJsonFields(body: []const u8) !ParsedJsonFields {
    var result = ParsedJsonFields{
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
        }
        if (s[i] == '"') return i;
    }
    return null;
}

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

test "completionFinishReason reports clamped generations as length" {
    try std.testing.expectEqual(FinishReason.length, completionFinishReason(256, 64, 64));
    try std.testing.expectEqual(FinishReason.length, completionFinishReason(32, 0, 0));
    try std.testing.expectEqual(FinishReason.stop, completionFinishReason(256, 64, 12));
    try std.testing.expectEqual(FinishReason.stop, completionFinishReason(64, 64, 64));
}

test "timingStats handles normal and empty decode windows" {
    const normal = timingStats(4, 2_000_000);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), normal.ms, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 2000.0), normal.tps, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), normal.ms_per_token, 0.0001);

    const empty = timingStats(0, 2_000_000);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), empty.ms, 0.0001);
    try std.testing.expectEqual(@as(f64, 0.0), empty.tps);
    try std.testing.expectEqual(@as(f64, 0.0), empty.ms_per_token);
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

/// Send content text through the streaming detector.
/// Errors propagate upward; callers use `catch return` to stop streaming on failure.
fn streamTextViaDetector(
    conn: *http.Connection,
    text: []const u8,
    req_id: []const u8,
    ts: i64,
    model_name: []const u8,
    detector: *tool_format.StreamingDetector,
    any_tool_call_emitted: *bool,
    tool_call_index: *u32,
    allocator: std.mem.Allocator,
    tools_active: bool,
) !void {
    // When the request didn't carry tools (or tool calling is disabled),
    // bypass the detector and stream bytes directly. This avoids the
    // detector's hold-on-'<' buffering, restoring mainline streaming
    // latency for the common no-tools path.
    if (!tools_active) return streamText(conn, text, req_id, ts, model_name);

    const fr = try detector.feed(text);
    switch (fr) {
        .emit_as_content => {
            const c = detector.takeContentDelta();
            if (c.len > 0) try streamText(conn, c, req_id, ts, model_name);
        },
        .hold => {},
        .tool_call_complete => {
            const c = detector.takeContentDelta();
            if (c.len > 0) try streamText(conn, c, req_id, ts, model_name);
            while (detector.takePendingToolCall()) |tc| {
                defer {
                    allocator.free(tc.id);
                    allocator.free(tc.name);
                    allocator.free(tc.arguments_json);
                }
                const ev = try formatStreamingToolCallChunk(allocator, req_id, ts, model_name, tool_call_index.*, tc.id, tc.name, tc.arguments_json);
                defer allocator.free(ev);
                try conn.writeSseEvent(ev);
                any_tool_call_emitted.* = true;
                tool_call_index.* += 1;
            }
        },
    }
}

fn formatStreamingToolCallChunk(
    allocator: std.mem.Allocator,
    req_id: []const u8,
    ts: i64,
    model_name: []const u8,
    tool_call_index: u32,
    id: []const u8,
    name: []const u8,
    arguments_json: []const u8,
) ![]u8 {
    var wb: std.ArrayList(u8) = .{};
    errdefer wb.deinit(allocator);

    try wb.writer(allocator).print(
        \\{{"id":"{s}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":{d},"id":"{s}","type":"function","function":{{"name":"{s}","arguments":"
    , .{ req_id, ts, model_name, tool_call_index, id, name });

    for (arguments_json) |c| {
        switch (c) {
            '"' => try wb.appendSlice(allocator, "\\\""),
            '\\' => try wb.appendSlice(allocator, "\\\\"),
            '\n' => try wb.appendSlice(allocator, "\\n"),
            '\r' => try wb.appendSlice(allocator, "\\r"),
            '\t' => try wb.appendSlice(allocator, "\\t"),
            else => try wb.append(allocator, c),
        }
    }

    try wb.appendSlice(allocator, "\"}}]},\"finish_reason\":null}]}");
    return wb.toOwnedSlice(allocator);
}

fn forcedToolArgumentTail(generated_text: []const u8) []const u8 {
    var out = trimTrailingChatArtifacts(generated_text);
    if (std.mem.indexOf(u8, out, "</tool_call>")) |idx| {
        out = out[0..idx];
    }
    if (std.mem.indexOf(u8, out, "<|im_end|>")) |idx| {
        out = out[0..idx];
    }
    return std.mem.trim(u8, out, " \t\r\n");
}

fn firstCompleteJsonObjectSlice(text: []const u8) ?[]const u8 {
    const start = std.mem.indexOfScalar(u8, text, '{') orelse return null;
    var depth: usize = 0;
    var in_string = false;
    var escaped = false;
    var i = start;
    while (i < text.len) : (i += 1) {
        const c = text[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }

        switch (c) {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                if (depth == 0) return null;
                depth -= 1;
                if (depth == 0) return text[start .. i + 1];
            },
            else => {},
        }
    }
    return null;
}

fn forcedToolArgumentsSlice(generated_text: []const u8) []const u8 {
    var out = forcedToolArgumentTail(generated_text);
    if (firstCompleteJsonObjectSlice(out)) |json_object| {
        return json_object;
    }

    if (out.len >= 2 and
        ((out[0] == '\'' and out[out.len - 1] == '\'') or
            (out[0] == '"' and out[out.len - 1] == '"')))
    {
        out = std.mem.trim(u8, out[1 .. out.len - 1], " \t\r\n");
        if (firstCompleteJsonObjectSlice(out)) |json_object| {
            return json_object;
        }
    }
    return "";
}

fn forcedToolArgumentsJsonAlloc(allocator: std.mem.Allocator, first_arg_name: ?[]const u8, generated_text: []const u8) ![]u8 {
    var merged: std.ArrayList(u8) = .{};
    defer merged.deinit(allocator);

    if (first_arg_name) |arg_name| {
        try merged.writer(allocator).print("{{\"{s}\":", .{arg_name});
    }
    try merged.appendSlice(allocator, forcedToolArgumentTail(generated_text));

    const args = forcedToolArgumentsSlice(merged.items);
    if (args.len == 0) return allocator.dupe(u8, "{}");

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, args, .{}) catch {
        return allocator.dupe(u8, "{}");
    };
    defer parsed.deinit();
    if (parsed.value != .object) return allocator.dupe(u8, "{}");

    return allocator.dupe(u8, args);
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
        .mistral => "mistral",
        .mamba => "mamba",
        .jamba => "jamba",
        .gemma => "gemma",
        .gpt_oss => "gpt-oss-20b",
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

test "shouldForceDisableThinking follows qwen catalog stability flags" {
    try std.testing.expect(!shouldForceDisableThinking(
        null,
        "/Users/test/Library/Caches/zinc/models/models/qwen36-35b-a3b-q4k-xl/model.gguf",
        "Qwen3.6-35B-A3B-UD-Q4_K_XL",
    ));
    try std.testing.expect(!shouldForceDisableThinking(
        "qwen35-9b-q4k-m",
        "/Users/test/Library/Caches/zinc/models/models/qwen35-9b-q4k-m/model.gguf",
        "Qwen3.5-9B-Q4_K_M",
    ));
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

test "lastCompleteUtf8End passes complete ASCII" {
    try std.testing.expectEqual(@as(usize, 5), lastCompleteUtf8End("hello"));
}

test "lastCompleteUtf8End passes complete multi-byte char" {
    // ⭐ U+2B50 = 0xE2 0xAD 0x90 (3 bytes), preceded by ASCII
    const s = "hi \xE2\xAD\x90";
    try std.testing.expectEqual(s.len, lastCompleteUtf8End(s));
}

test "lastCompleteUtf8End trims partial 4-byte emoji" {
    // 🔧 U+1F527 = 0xF0 0x9F 0x94 0xA7. Cut at each prefix length 1..3 — should
    // trim back to before the lead byte. Full sequence passes through.
    const full = "abc\xF0\x9F\x94\xA7";
    try std.testing.expectEqual(full.len, lastCompleteUtf8End(full));
    try std.testing.expectEqual(@as(usize, 3), lastCompleteUtf8End(full[0..4])); // F0
    try std.testing.expectEqual(@as(usize, 3), lastCompleteUtf8End(full[0..5])); // F0 9F
    try std.testing.expectEqual(@as(usize, 3), lastCompleteUtf8End(full[0..6])); // F0 9F 94
}

test "lastCompleteUtf8End trims partial 3-byte sequence" {
    // ⭐ split mid-codepoint
    const full = "ok \xE2\xAD\x90";
    try std.testing.expectEqual(@as(usize, 3), lastCompleteUtf8End(full[0..4])); // E2
    try std.testing.expectEqual(@as(usize, 3), lastCompleteUtf8End(full[0..5])); // E2 AD
    try std.testing.expectEqual(full.len, lastCompleteUtf8End(full));
}

test "lastCompleteUtf8End empty input" {
    try std.testing.expectEqual(@as(usize, 0), lastCompleteUtf8End(""));
}

test "lastCompleteUtf8End streams emoji byte-by-byte without dropping bytes" {
    // Simulate the streaming server's flush boundary: a 4-byte emoji arrives
    // one byte at a time; each call must hold back the partial bytes until
    // the next chunk completes the codepoint. Concatenating the emitted
    // chunks must reproduce the original bytes exactly.
    const full = "x\xF0\x9F\x94\xA7y";
    var emitted: std.ArrayList(u8) = .{};
    defer emitted.deinit(std.testing.allocator);

    var sent: usize = 0;
    var len: usize = 0;
    while (len <= full.len) : (len += 1) {
        const safe = lastCompleteUtf8End(full[0..len]);
        if (safe > sent) {
            try emitted.appendSlice(std.testing.allocator, full[sent..safe]);
            sent = safe;
        }
    }
    try std.testing.expectEqualSlices(u8, full, emitted.items);
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
    const prompt = try buildChatPrompt(std.testing.allocator, &tok, &roles, &contents, null, false, &.{}, .auto, &buf);
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

test "normalizeStructuredAssistantOutput strips Harmony analysis when thinking is disabled" {
    var tok = makeTestTokenizer("<|start|>assistant<|message|>");
    defer tok.token_to_id.deinit();

    const raw =
        "<|channel|>analysis<|message|>We need to answer briefly.<|end|>" ++
        "<|start|>assistant<|channel|>final<|message|>Paris<|return|>";
    var buf: [512]u8 = undefined;
    const cleaned = try normalizeStructuredAssistantOutput(&tok, raw, false, &buf);
    try std.testing.expectEqualStrings("Paris", cleaned);
}

test "normalizeStructuredAssistantOutput preserves Harmony analysis when thinking is enabled" {
    var tok = makeTestTokenizer("<|start|>assistant<|message|>");
    defer tok.token_to_id.deinit();

    const raw =
        "<|channel|>analysis<|message|>We need to answer briefly.<|end|>" ++
        "<|start|>assistant<|channel|>final<|message|>Paris<|return|>";
    var buf: [512]u8 = undefined;
    const cleaned = try normalizeStructuredAssistantOutput(&tok, raw, true, &buf);
    try std.testing.expectEqualStrings("<think>\nWe need to answer briefly.\n</think>\nParis", cleaned);
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

test "pendingGeneratedText clamps stale sent cursor" {
    const text = "abcdef";
    var sent_len: usize = 9;
    const pending = pendingGeneratedText(text, &sent_len, 3);
    try std.testing.expectEqual(@as(usize, 3), sent_len);
    try std.testing.expectEqualStrings("", pending);
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
    const prompt = try buildChatPrompt(std.testing.allocator, &tok, &roles, &contents, null, false, &.{}, .auto, &buf);
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
    const prompt = try buildChatPrompt(std.testing.allocator, &tok, &roles, &contents, true, false, &.{}, .auto, &buf);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>user\nhello<|im_end|>\n") != null);
    try std.testing.expect(std.mem.endsWith(u8, prompt, "<|im_start|>assistant\n<think>\n"));
    try std.testing.expect(std.mem.indexOf(u8, prompt, "</think>") == null);
}

test "tool-calling requests skip qwen thinking scaffold" {
    try std.testing.expect(shouldSkipThinkingTemplateForRequest(1, .auto));
    try std.testing.expect(shouldSkipThinkingTemplateForRequest(1, .required));
    try std.testing.expect(!shouldSkipThinkingTemplateForRequest(1, .none));
    try std.testing.expect(!shouldSkipThinkingTemplateForRequest(0, .auto));

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
    const contents = [_][]const u8{"call a tool"};
    const tools = [_]tool_format.ToolDefinition{.{
        .name = "read",
        .description = "Read a file.",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"filePath\":{\"type\":\"string\"}}}",
    }};
    var buf: [4096]u8 = undefined;
    const prompt = try buildChatPrompt(std.testing.allocator, &tok, &roles, &contents, null, true, &tools, .auto, &buf);
    try std.testing.expect(std.mem.endsWith(u8, prompt, "<|im_start|>assistant\n"));
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<think>") == null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<tool_call>") != null);
}

test "force-disabled qwen thinking keeps closed generation scaffold" {
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
    const prompt = try buildChatPrompt(std.testing.allocator, &tok, &roles, &contents, false, false, &.{}, .auto, &buf);
    try std.testing.expect(std.mem.endsWith(u8, prompt, "<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}

test "required single-tool prompt prefills the tool call arguments slot" {
    var tok = makeTestTokenizer(null);
    defer tok.token_to_id.deinit();

    const roles = [_][]const u8{ "system", "user" };
    const contents = [_][]const u8{ "You are a coding agent.", "Write hello." };
    const tools = [_]tool_format.ToolDefinition{.{
        .name = "write",
        .description = "Write a file.",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"filePath\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}",
    }};
    var buf: [4096]u8 = undefined;
    const prompt = try buildChatPrompt(std.testing.allocator, &tok, &roles, &contents, null, true, &tools, .required, &buf);
    try std.testing.expect(std.mem.endsWith(u8, prompt, "<|im_start|>assistant\n<tool_call>\n{\"name\":\"write\",\"arguments\":{\"filePath\":"));
}

test "firstToolArgumentName prefers required key then properties key" {
    try std.testing.expectEqualStrings(
        "content",
        firstToolArgumentName("{\"type\":\"object\",\"properties\":{\"filePath\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},\"required\":[\"content\",\"filePath\"]}").?,
    );
    try std.testing.expectEqualStrings(
        "filePath",
        firstToolArgumentName("{\"type\":\"object\",\"properties\":{\"filePath\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}").?,
    );
}

test "forcedToolArgumentsSlice extracts the generated JSON object" {
    try std.testing.expectEqualStrings(
        "{\"filePath\":\"/tmp/a\",\"content\":\"hello\"}",
        forcedToolArgumentsSlice("  '{\"filePath\":\"/tmp/a\",\"content\":\"hello\"}'\n</tool_call>\n<|im_end|>"),
    );
    try std.testing.expectEqualStrings(
        "{\"path\":\"/tmp/a\",\"content\":\"hello\"}",
        forcedToolArgumentsSlice("{\"path\":\"/tmp/a\",\"content\":\"hello\"}</tool_call> trailing"),
    );
    try std.testing.expectEqualStrings(
        "{\"filePath\":\"/tmp/a\",\"content\":\"hello\"}",
        forcedToolArgumentsSlice("{\"filePath\":\"/tmp/a\",\"content\":\"hello\"}}</tool_call> trailing"),
    );
}

test "forcedToolArgumentsJsonAlloc rebuilds prefilled first argument" {
    const args = try forcedToolArgumentsJsonAlloc(std.testing.allocator, "filePath", "\"/tmp/a\",\"content\":\"hello\"}}</tool_call>");
    defer std.testing.allocator.free(args);
    try std.testing.expectEqualStrings("{\"filePath\":\"/tmp/a\",\"content\":\"hello\"}", args);

    const invalid = try forcedToolArgumentsJsonAlloc(std.testing.allocator, "filePath", "}</tool_call>");
    defer std.testing.allocator.free(invalid);
    try std.testing.expectEqualStrings("{}", invalid);
}

test "buildChatPrompt succeeds for large Goose-style tool prompts" {
    var tok = makeTestTokenizer(null);
    defer tok.token_to_id.deinit();

    const roles = [_][]const u8{
        "system",
        "user",
    };
    const contents = [_][]const u8{
        "You are a general-purpose AI agent called goose, created by AAIF (Agentic AI Foundation).\n" ++
            "goose is being developed as an open-source software project.\n\n" ++
            "# Extensions\n\n" ++
            "Extensions provide additional tools and context from different data sources and applications.\n" ++
            "You can dynamically enable or disable extensions as needed to help complete tasks.\n\n" ++
            "Because you dynamically load extensions, your conversation history may refer\n" ++
            "to interactions with extensions that are not currently active. The currently\n" ++
            "active extensions are below. Each of these extensions provides tools that are\n" ++
            "in your tool specification.\n\n" ++
            "## developer\n\n" ++
            "### Instructions\n" ++
            "Use the developer extension to build software and operate a terminal.\n\n" ++
            "Make sure to use the tools efficiently and minimize unnecessary turns.\n",
        "hello",
    };

    const tools = [_]tool_format.ToolDefinition{
        .{
            .name = "analyze",
            .description = "Analyze code structure in 3 modes.",
            .parameters_json =
            \\{"$schema":"https://json-schema.org/draft/2020-12/schema","title":"AnalyzeParams","type":"object","properties":{"path":{"description":"File or directory path to analyze","type":"string"},"focus":{"description":"Symbol name to focus on","type":["string","null"],"default":null},"max_depth":{"description":"Directory recursion depth limit","type":"integer","minimum":0,"default":3},"follow_depth":{"description":"Call graph traversal depth","type":"integer","minimum":0,"default":2},"force":{"description":"Allow large outputs without size warning","type":"boolean","default":false}},"required":["path"]}
            ,
        },
        .{
            .name = "delegate",
            .description = "Delegate a task to a subagent that runs independently with its own context.",
            .parameters_json =
            \\{"type":"object","properties":{"instructions":{"type":"string"},"source":{"type":"string"},"parameters":{"type":"object","additionalProperties":true},"extensions":{"type":"array","items":{"type":"string"}},"provider":{"type":"string"},"model":{"type":"string"},"temperature":{"type":"number"},"max_turns":{"type":"integer","minimum":1},"async":{"type":"boolean","default":false}},"required":[]}
            ,
        },
        .{
            .name = "shell",
            .description = "Execute a shell command in the current dir.",
            .parameters_json =
            \\{"$schema":"https://json-schema.org/draft/2020-12/schema","title":"ShellParams","type":"object","properties":{"command":{"type":"string"},"timeout_secs":{"type":["integer","null"],"minimum":0,"default":null}},"required":["command"]}
            ,
        },
    };

    const estimated = estimateChatPromptBytes(&roles, &contents, false, &tools);
    const prompt_buf = try std.testing.allocator.alloc(u8, estimated);
    defer std.testing.allocator.free(prompt_buf);

    const prompt = buildChatPrompt(std.testing.allocator, &tok, &roles, &contents, null, false, &tools, .auto, prompt_buf) catch |err| {
        if (err == error.BufferTooSmall) {
            std.debug.print("estimated={d}\n", .{estimated});
        }
        return err;
    };

    try std.testing.expect(std.mem.indexOf(u8, prompt, "<tools>") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "delegate") != null);
    try std.testing.expect(std.mem.endsWith(u8, prompt, "<|im_start|>assistant\n"));
}

test "persistentChatHistoryLen drops trailing OpenCode proxy hints" {
    const roles = [_][]const u8{ "system", "user", "system", "system" };
    const contents = [_][]const u8{
        "be direct",
        "fix the project",
        "Tool path guard: use only /tmp/app/src/cart.mjs",
        "OpenCode continuation guard: edit the source file next",
    };

    try std.testing.expectEqual(@as(usize, 2), persistentChatHistoryLen(&roles, &contents));
}

test "persistentChatHistoryLen keeps non-trailing proxy-looking history" {
    const roles = [_][]const u8{ "system", "user", "assistant" };
    const contents = [_][]const u8{
        "Tool path guard: this was part of the original prompt",
        "fix the project",
        "done",
    };

    try std.testing.expectEqual(@as(usize, 3), persistentChatHistoryLen(&roles, &contents));
}

test "buildChatTranscriptPrompt renders tools without generation suffix" {
    var tok = makeTestTokenizer(null);
    defer tok.token_to_id.deinit();

    const roles = [_][]const u8{ "system", "user", "assistant" };
    const contents = [_][]const u8{ "be precise", "read package.json", "I will inspect it." };
    const tools = [_]tool_format.ToolDefinition{.{
        .name = "read",
        .description = "Read a file.",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"filePath\":{\"type\":\"string\"}}}",
    }};

    var buf: [4096]u8 = undefined;
    const prompt = try buildChatTranscriptPrompt(std.testing.allocator, &tok, &roles, &contents, false, false, &tools, .auto, &buf);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "# Tools") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "\"name\": \"read\"") != null);
    try std.testing.expect(!std.mem.endsWith(u8, prompt, "<|im_start|>assistant\n"));
}

test "formatStreamingToolCallChunk handles large arguments payloads" {
    var big_args: [24000]u8 = undefined;
    @memset(&big_args, 'a');

    const prefix = "{\"path\":\"";
    const suffix = "\"}";
    var args: std.ArrayList(u8) = .{};
    defer args.deinit(std.testing.allocator);
    try args.appendSlice(std.testing.allocator, prefix);
    try args.appendSlice(std.testing.allocator, &big_args);
    try args.appendSlice(std.testing.allocator, suffix);

    const chunk = try formatStreamingToolCallChunk(
        std.testing.allocator,
        "chatcmpl-test",
        123,
        "qwen36-35b-a3b-q4k-xl",
        0,
        "call_0",
        "shell",
        args.items,
    );
    defer std.testing.allocator.free(chunk);

    try std.testing.expect(std.mem.indexOf(u8, chunk, "\"tool_calls\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, chunk, "\"name\":\"shell\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, chunk, "\"arguments\":\"{\\\"path\\\":\\\"") != null);
    try std.testing.expect(chunk.len > 24000);
}

test "formatStreamingToolCallChunk emits valid JSON" {
    const chunk = try formatStreamingToolCallChunk(
        std.testing.allocator,
        "chatcmpl-test",
        123,
        "Qwen3.6-35B-A3B",
        0,
        "call_0",
        "tree",
        "{\"path\":\"/home/f44/dev/stuff/pwmedia\",\"depth\":3}",
    );
    defer std.testing.allocator.free(chunk);

    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, chunk, .{});
    defer parsed.deinit();

    try std.testing.expect(parsed.value == .object);
    const choices = parsed.value.object.get("choices") orelse return error.TestUnexpectedResult;
    try std.testing.expect(choices == .array);
    try std.testing.expectEqual(@as(usize, 1), choices.array.items.len);

    const choice0 = choices.array.items[0];
    try std.testing.expect(choice0 == .object);
    const delta = choice0.object.get("delta") orelse return error.TestUnexpectedResult;
    try std.testing.expect(delta == .object);
    const tool_calls = delta.object.get("tool_calls") orelse return error.TestUnexpectedResult;
    try std.testing.expect(tool_calls == .array);
    try std.testing.expectEqual(@as(usize, 1), tool_calls.array.items.len);

    const tool0 = tool_calls.array.items[0];
    try std.testing.expect(tool0 == .object);
    const function = tool0.object.get("function") orelse return error.TestUnexpectedResult;
    try std.testing.expect(function == .object);
    const arguments = function.object.get("arguments") orelse return error.TestUnexpectedResult;
    try std.testing.expect(arguments == .string);
    try std.testing.expectEqualStrings("{\"path\":\"/home/f44/dev/stuff/pwmedia\",\"depth\":3}", arguments.string);
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
    try std.testing.expect(std.mem.indexOf(u8, parsed.contents[0], "Answer directly") != null);
    try std.testing.expectEqualStrings("user", parsed.roles[1]);
}

test "dropInjectedDefaultSystemForGemma removes only synthetic guidance" {
    const body =
        \\{"messages":[{"role":"user","content":"hello"}]}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();

    try std.testing.expect(parsed.injected_default_system);
    dropInjectedDefaultSystemForGemma(&parsed, true);

    try std.testing.expect(!parsed.injected_default_system);
    try std.testing.expectEqual(@as(usize, 1), parsed.roles.len);
    try std.testing.expectEqualStrings("user", parsed.roles[0]);
    try std.testing.expectEqualStrings("hello", parsed.contents[0]);
}

test "dropInjectedDefaultSystemForGemma preserves explicit system guidance" {
    const body =
        \\{"messages":[{"role":"system","content":"be terse"},{"role":"user","content":"hello"}]}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();

    try std.testing.expect(!parsed.injected_default_system);
    dropInjectedDefaultSystemForGemma(&parsed, true);

    try std.testing.expectEqual(@as(usize, 2), parsed.roles.len);
    try std.testing.expectEqualStrings("system", parsed.roles[0]);
    try std.testing.expectEqualStrings("be terse", parsed.contents[0]);
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
        .{ .role = "", .content = .{ .text = "ignored" } },
        .{ .role = "user", .content = .{ .text = "" } },
        .{ .role = "user", .content = .{ .text = "hello" } },
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

test "parseChatRequest accepts content as a single text block array (OpenAI multimodal shape)" {
    const body =
        \\{"messages":[{"role":"user","content":[{"type":"text","text":"hello world"}]}]}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();

    // system prompt + user message
    try std.testing.expectEqual(@as(usize, 2), parsed.roles.len);
    try std.testing.expectEqualStrings("user", parsed.roles[1]);
    try std.testing.expectEqualStrings("hello world", parsed.contents[1]);
}

test "parseChatRequest concatenates multiple text blocks in order" {
    const body =
        \\{"messages":[{"role":"user","content":[{"type":"text","text":"first "},{"type":"text","text":"second"}]}]}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();

    try std.testing.expectEqualStrings("user", parsed.roles[1]);
    try std.testing.expectEqualStrings("first second", parsed.contents[1]);
}

test "parseChatRequest skips non-text blocks without failing" {
    // image_url and unknown block types should be silently dropped, not error.
    const body =
        \\{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://x"}},{"type":"text","text":"caption"}]}]}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();

    try std.testing.expectEqualStrings("caption", parsed.contents[1]);
}

test "parseChatRequest accepts null content (assistant tool-call message)" {
    setToolCallingForTest(true);
    defer resetToolCallingForTest();
    // Assistant turn with content: null and tool_calls is preserved in history
    // — the model needs to see the previous tool invocations rendered as
    // <tool_call> blocks so subsequent tool_results have context. Pre-tool-
    // calling the assistant turn used to be skipped here; now it's rendered.
    const body =
        \\{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"user","content":"follow up"}]}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();

    // system + user + assistant(rendered tool_calls) + user
    try std.testing.expectEqual(@as(usize, 4), parsed.roles.len);
    try std.testing.expectEqualStrings("system", parsed.roles[0]);
    try std.testing.expectEqualStrings("user", parsed.roles[1]);
    try std.testing.expectEqualStrings("q", parsed.contents[1]);
    try std.testing.expectEqualStrings("assistant", parsed.roles[2]);
    try std.testing.expect(std.mem.indexOf(u8, parsed.contents[2], "<tool_call>") != null);
    try std.testing.expect(std.mem.indexOf(u8, parsed.contents[2], "\"name\": \"f\"") != null);
    try std.testing.expectEqualStrings("user", parsed.roles[3]);
    try std.testing.expectEqualStrings("follow up", parsed.contents[3]);
}

test "parseChatRequest parses tools and tool_choice" {
    setToolCallingForTest(true);
    defer resetToolCallingForTest();
    const body =
        \\{"messages":[{"role":"user","content":"what time is it?"}],"tools":[{"type":"function","function":{"name":"get_time","description":"Get current time","parameters":{"type":"object"}}}],"tool_choice":"auto"}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 1), parsed.tools.len);
    try std.testing.expectEqualStrings("get_time", parsed.tools[0].name);
    try std.testing.expectEqualStrings("Get current time", parsed.tools[0].description);
    try std.testing.expectEqual(ToolChoice.auto, parsed.tool_choice);

    const required_body =
        \\{"messages":[{"role":"user","content":"what time is it?"}],"tools":[{"type":"function","function":{"name":"get_time","description":"Get current time","parameters":{"type":"object"}}}],"tool_choice":"required"}
    ;
    var required = try parseChatRequest(std.testing.allocator, required_body);
    defer required.deinit();
    try std.testing.expectEqual(ToolChoice.required, required.tool_choice);
}

test "parseChatRequest tool_choice none suppresses tools" {
    setToolCallingForTest(true);
    defer resetToolCallingForTest();
    const body =
        \\{"messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"fn1","description":"d","parameters":null}}],"tool_choice":"none"}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();

    try std.testing.expectEqual(ToolChoice.none, parsed.tool_choice);
    // tools array is still populated; callers use tool_choice to decide whether to render them
    try std.testing.expectEqual(@as(usize, 1), parsed.tools.len);
}

test "parseChatRequest replays assistant tool_calls as tool_call blocks" {
    setToolCallingForTest(true);
    defer resetToolCallingForTest();
    const body =
        \\{"messages":[{"role":"user","content":"call it"},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"get_time","arguments":"{}"}}]},{"role":"tool","content":"12:00","tool_call_id":"c1"},{"role":"user","content":"thanks"}]}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();

    // Expect: system + user + assistant(tool_call) + tool + user = 5 messages
    // (default system is prepended since no system message)
    var found_tool_call_block = false;
    for (parsed.contents) |c| {
        if (std.mem.indexOf(u8, c, "<tool_call>") != null) {
            found_tool_call_block = true;
            try std.testing.expect(std.mem.indexOf(u8, c, "get_time") != null);
        }
    }
    try std.testing.expect(found_tool_call_block);
}

test "parseChatRequest preserves assistant text before historical tool_calls" {
    setToolCallingForTest(true);
    defer resetToolCallingForTest();
    const body =
        \\{"messages":[{"role":"user","content":"inspect"},{"role":"assistant","content":"I'll read the source first.","tool_calls":[{"id":"c1","type":"function","function":{"name":"read","arguments":"{\"filePath\":\"/tmp/app/src/cart.mjs\"}"}}]},{"role":"tool","content":"export function subtotalCents() {}","tool_call_id":"c1"}]}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();

    try std.testing.expectEqualStrings("assistant", parsed.roles[2]);
    try std.testing.expect(std.mem.indexOf(u8, parsed.contents[2], "I'll read the source first.") != null);
    try std.testing.expect(std.mem.indexOf(u8, parsed.contents[2], "<tool_call>") != null);
    try std.testing.expect(std.mem.indexOf(u8, parsed.contents[2], "\"name\": \"read\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, parsed.contents[2], "/tmp/app/src/cart.mjs") != null);
}

// ──────────────────────────────────────────────────────────────────
// Gate-off regression tests for ZINC_TOOL_CALLING.
//
// These tests pin the "default behavior is mainline-identical when
// ZINC_TOOL_CALLING is unset" guarantee. If someone refactors the
// gate and accidentally enables tool-calling unconditionally — or
// disables it when it should be on — these will catch it.
// ──────────────────────────────────────────────────────────────────

test "parseChatRequest with gate off ignores request tools field" {
    setToolCallingForTest(false);
    defer resetToolCallingForTest();
    const body =
        \\{"messages":[{"role":"user","content":"go"}],"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":null}}]}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 0), parsed.tools.len);
}

test "parseChatRequest with gate off skips assistant turn that has only tool_calls" {
    setToolCallingForTest(false);
    defer resetToolCallingForTest();
    // Same body as the gate-on null-content test, but the assistant turn
    // must be dropped instead of rendered, so we expect 3 roles not 4.
    const body =
        \\{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"user","content":"hi"}]}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 3), parsed.roles.len);
    try std.testing.expectEqualStrings("system", parsed.roles[0]);
    try std.testing.expectEqualStrings("user", parsed.roles[1]);
    try std.testing.expectEqualStrings("user", parsed.roles[2]);
}

test "parseChatRequest with gate off forces tool_choice to auto" {
    setToolCallingForTest(false);
    defer resetToolCallingForTest();
    const body =
        \\{"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":null}}],"tool_choice":"none"}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();
    try std.testing.expectEqual(ToolChoice.auto, parsed.tool_choice);
    try std.testing.expectEqual(@as(usize, 0), parsed.tools.len);
}

test "toolCallingEnabled cache state machine" {
    const prev = tool_calling_state.load(.acquire);
    defer tool_calling_state.store(prev, .release);

    // Forced-on cache → returns true, cache unchanged.
    tool_calling_state.store(1, .release);
    try std.testing.expect(toolCallingEnabled());
    try std.testing.expectEqual(@as(i8, 1), tool_calling_state.load(.acquire));

    // Forced-off cache → returns false, cache unchanged.
    tool_calling_state.store(0, .release);
    try std.testing.expect(!toolCallingEnabled());
    try std.testing.expectEqual(@as(i8, 0), tool_calling_state.load(.acquire));

    // Uncached → probes env, then caches the result. We don't control the
    // env here, but the post-call cache must be 0 or 1 (no longer -1).
    tool_calling_state.store(-1, .release);
    _ = toolCallingEnabled();
    try std.testing.expect(tool_calling_state.load(.acquire) >= 0);
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

test "historyAssistantContentWithTools keeps assistant tool calls in cached history" {
    var qwen_tok = makeTestTokenizer(null);
    defer qwen_tok.token_to_id.deinit();

    var transport_buf: [512]u8 = undefined;
    var tool_history_buf: std.ArrayList(u8) = .{};
    defer tool_history_buf.deinit(std.testing.allocator);
    const cached = try historyAssistantContentWithTools(
        &qwen_tok,
        tool_format.chatmlToolFormat(),
        true,
        "I'll read it.\n<tool_call>\n{\"name\":\"read\",\"arguments\":{\"filePath\":\"/tmp/app/src/cart.mjs\"}}\n</tool_call>\n",
        &transport_buf,
        &tool_history_buf,
        std.testing.allocator,
    );

    try std.testing.expect(std.mem.indexOf(u8, cached, "I'll read it.") != null);
    try std.testing.expect(std.mem.indexOf(u8, cached, "<tool_call>") != null);
    try std.testing.expect(std.mem.indexOf(u8, cached, "\"name\": \"read\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, cached, "/tmp/app/src/cart.mjs") != null);
}

test "stripThinkingForDisabledResponse removes complete think block and keeps answer" {
    var buf: [256]u8 = undefined;
    const stripped = try stripThinkingForDisabledResponse("<think>\n17 * 24 = 408\n</think>\n\n408", &buf);
    try std.testing.expectEqualStrings("408", stripped);
}

test "stripThinkingForDisabledResponse promotes think-only answer when no final answer arrived" {
    var buf: [256]u8 = undefined;
    const stripped = try stripThinkingForDisabledResponse("<think>\nZig is a systems programming language focused on explicit control.\n</think>\n", &buf);
    try std.testing.expectEqualStrings("Zig is a systems programming language focused on explicit control.", stripped);
}

test "stripThinkingForDisabledResponse truncates partial think block after visible answer" {
    var buf: [256]u8 = undefined;
    const stripped = try stripThinkingForDisabledResponse("Hello! How can I help you today?\"\n\n<think>\nThinking Process:\n", &buf);
    const cleaned = trimTrailingChatArtifacts(stripped);
    try std.testing.expectEqualStrings("Hello! How can I help you today?", cleaned);
}

test "stripThinkingForDisabledResponse preserves text outside think tags" {
    var buf: [256]u8 = undefined;
    const stripped = try stripThinkingForDisabledResponse("Visible text\n<think>\nhidden\n</think>\n\nMore text", &buf);
    try std.testing.expectEqualStrings("Visible text\nMore text", stripped);
}

test "stripThinkingForDisabledResponse falls back to partial thinking body when no answer arrived yet" {
    var buf: [256]u8 = undefined;
    const stripped = try stripThinkingForDisabledResponse("<think>\nZig is a systems programming language focused on explicit control.", &buf);
    try std.testing.expectEqualStrings("Zig is a systems programming language focused on explicit control.", stripped);
}

test "hasDanglingTrailingQuote treats standalone trailing quote as dangling" {
    try std.testing.expect(hasDanglingTrailingQuote("\""));
    try std.testing.expect(hasDanglingTrailingQuote("answer\""));
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
