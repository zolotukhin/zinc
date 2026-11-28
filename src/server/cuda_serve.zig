//! Shared ROCm/CUDA batched serving worker.
//! @section API Server
//!
//! The selected C shim supplies AMD HIP or NVIDIA CUDA execution; the
//! `cuda_serve` module name is shared by both native backends.
//!
//! One GPU **worker thread** runs the continuous-batching loop
//! (admit → prefill → `decodeBatch` → evict); many transport (HTTP) handler
//! threads `submit` requests concurrently and **stream each request's own tokens
//! incrementally** as the worker produces them. This is the threading model
//! proven token-identical to N isolated single-sequence runs by the `dbg_cuda
//! serve` harness (3a); 3b factors it into a reusable engine and adds the
//! per-token streaming registry the HTTP/SSE transport needs.
//!
//! ADDITIVE: the production single-sequence `decodeStep`/`prefillBatched` path is
//! untouched. The engine REUSES `Scheduler` + `ForwardGemma.decodeBatch` (proven
//! in increments 1+2); the only genuinely-new code here is the cross-thread
//! result registry + the GPU worker loop.
//!
//! Thread-safety model (identical to the 3a proof):
//!   * ALL GPU work (`decodeBatch`, prefill) runs ONLY on the worker thread. The
//!     CUDA shim rebinds the context per call (`cuCtxSetCurrent` at every entry),
//!     so a single GPU-owning thread needs no extra ceremony.
//!   * Cross-thread mutable state = the scheduler `pending` FIFO (handlers append
//!     via `enqueue`; worker drains via `admitNext`) + the per-request channel
//!     registry. Both are guarded by ONE `mutex`. Slot state (prefill / decode /
//!     append / release) and the scratch slices are worker-only and lock-free.
//!   * Each sequence's tokens depend only on its own slot KV + position (proven
//!     isolated in increment 1), so the nondeterministic admit/interleave ORDER
//!     across handler threads cannot change any sequence's output.

const std = @import("std");
const host_io = @import("host_io");
const scheduler = @import("../scheduler/scheduler.zig");
const forwardgemma = @import("../compute/forward_cuda_gemma.zig");
const is_rocm = std.mem.eql(u8, @import("build_options").backend, "rocm");
const Inkling = if (is_rocm) @import("../compute/forward_inkling.zig").ForwardInkling else noreturn;
const forwardcuda = @import("../compute/forward_cuda.zig");
const prefill_policy = @import("../compute/prefill_policy.zig");
const shim = @import("../cuda/c.zig").shim;
const mtp_policy = @import("mtp_policy.zig");

pub const MtpCapability = mtp_policy.Capability;
pub const SubmitOptions = mtp_policy.SubmitOptions;
pub const MtpResult = mtp_policy.Result;

const log = std.log.scoped(.cuda_serve);

/// One line per request, in the same shape the Vulkan server emits, so the
/// benchmark harness records acceptance for ROCm server runs too
/// (tools/performance_suite.mjs parseZincSpeculative matches this prefix).
fn logMtpAcceptance(drafted: u64, accepted: u64, cycles: u64) void {
    const pct = if (drafted > 0)
        100.0 * @as(f64, @floatFromInt(accepted)) / @as(f64, @floatFromInt(drafted))
    else
        0.0;
    log.info("NextN/MTP: request accepted {d}/{d} draft tokens ({d:.1}%) over {d} cycles", .{ accepted, drafted, pct, cycles });
}

/// Architecture-dispatched GPU forward held by the serving engine. gemma4 dense
/// (`ForwardGemma`) and the qwen35/36 hybrid-SSM family (`ForwardCuda`) expose the
/// SAME batched serving primitives — `decodeBatch(tokens,positions,slots,out)` and
/// per-sequence slot-state alloc — only under different method names, so the engine
/// drives EITHER through this thin union. Effort 28 increment 4 (qwen serving): the
/// Scheduler / ReqChannel / worker threading are model-agnostic; only these calls
/// differ. The forward is owned by the caller (its frame outlives the engine).
pub const Forward = union(enum) {
    gemma: *forwardgemma.ForwardGemma,
    qwen: *forwardcuda.ForwardCuda,
    inkling: if (is_rocm) *Inkling else noreturn,

    fn checkExecution(self: Forward) !void {
        const ctx = switch (self) {
            .gemma => |g| g.ctx,
            .qwen => |q| q.ctx,
            .inkling => |model| if (comptime is_rocm) model.ctx else unreachable,
        };
        if (shim.cuda_execution_failed(ctx) != 0) return error.GpuExecutionFailed;
    }

    fn computedPrefillTokens(self: Forward, total: usize) usize {
        return total -| switch (self) {
            .gemma, .inkling => @as(usize, 0),
            .qwen => |q| q.last_prefill_cached,
        };
    }

    fn allocSlots(self: Forward, nslots: u32, slot_ctx: u32) !void {
        switch (self) {
            .gemma => |g| try g.allocSlotKv(nslots, slot_ctx),
            .qwen => |q| try q.allocSlotState(nslots, slot_ctx),
            .inkling => |model| {
                if (comptime !is_rocm) unreachable;
                // Inkling's forward pass already owns one fixed-size sequence.
                if (nslots != 1 or slot_ctx != model.max_ctx) return error.InvalidParallelism;
            },
        }
    }
    fn freeSlots(self: Forward) void {
        switch (self) {
            .gemma => |g| g.freeSlotKv(),
            .qwen => |q| q.freeSlotState(),
            // ForwardInkling.deinit releases the single sequence's buffers.
            .inkling => {},
        }
    }
    fn decodeBatch(self: Forward, tokens: []const u32, positions: []const u32, slots: []const u32, out: []u32) !void {
        try self.checkExecution();
        switch (self) {
            .gemma => |g| try g.decodeBatch(tokens, positions, slots, out),
            .qwen => |q| try q.decodeBatch(tokens, positions, slots, out),
            .inkling => |model| if (comptime is_rocm) try model.decodeBatch(tokens, positions, slots, out) else unreachable,
        }
        try self.checkExecution();
    }
    /// Clear a reused slot's accumulated state before a new request prefills into
    /// it. No-op for gemma (attention-only, position-indexed KV overwritten on
    /// prefill); zeros the slot's SSM conv ring + recurrent state for qwen.
    fn resetSlot(self: Forward, slot: u32) !void {
        try self.checkExecution();
        switch (self) {
            .gemma => {},
            .qwen => |q| try q.resetSlot(slot),
            .inkling => |model| if (comptime is_rocm) try model.resetSlot(slot) else unreachable,
        }
        try self.checkExecution();
    }
    /// Batched prefill that writes state into the slot's per-sequence region.
    /// Falls back to per-token decodeBatch when unavailable (gemma or old build).
    fn prefillSlot(self: Forward, tokens: []const u32, slot: u32) !u32 {
        try self.checkExecution();
        const tok = switch (self) {
            .gemma => |g| try g.prefillBatchedSlot(tokens, slot),
            .qwen => |q| try q.prefillBatchedSlot(tokens, slot),
            .inkling => |model| if (comptime is_rocm) try model.prefillBatchedSlot(tokens, slot) else unreachable,
        };
        try self.checkExecution();
        return tok;
    }
};

/// Per-request published-token channel. The handler thread that submitted the
/// request OWNS this struct (it lives on the handler's stack/frame); the worker
/// appends generated tokens + flips `done`/`failed` under the engine mutex and
/// broadcasts. Registered by request id in `ServeEngine.registry`. ALL fields are
/// touched only under `ServeEngine.mutex` (the worker may realloc `tokens` while
/// the handler drains it, so the handler must copy out under the same lock).
pub const ReqChannel = struct {
    /// Tokens generated so far (worker appends under lock; `nextChunk` copies out).
    tokens: std.ArrayList(u32) = .empty,
    /// How many tokens the handler has already drained via `nextChunk`.
    consumed: usize = 0,
    /// True once the request finished (EOS or max_tokens) — no more tokens coming.
    done: bool = false,
    /// True if generation errored on the worker (GPU/alloc failure).
    failed: bool = false,
    /// Transport disconnected or reached a stop sequence. The worker observes
    /// this under the mutex and releases the slot at the next token boundary.
    cancelled: bool = false,
    /// Written before done/failed is published; read after the channel drains.
    mtp: MtpResult = .{},
    /// Per-request device timings. The worker records these before publishing
    /// completion; the HTTP handler reads them after `finished` is observed.
    prefill_tokens: u64 = 0,
    prefill_wall_ns: u64 = 0,
    decode_tokens: u64 = 0,
    decode_wall_ns: u64 = 0,
};

/// Result of `nextChunk`: how many fresh tokens were copied into the caller's
/// buffer and whether the stream is now fully drained.
pub const Chunk = struct {
    n: usize,
    finished: bool,
    failed: bool,
};

pub const ServeEngine = struct {
    allocator: std.mem.Allocator,
    fwd: Forward,
    sched: scheduler.Scheduler,
    /// Stop token id (the tokenizer's EOS for real serving; overridable for the gate).
    eos: u32,
    /// Optional model-specific end-of-turn token (ATEM `<|eot|>`).
    eot: ?u32,
    slot_ctx: u32,
    /// Immutable after initialization, including successful context preparation.
    mtp_capability: MtpCapability = .{},

    mutex: host_io.Mutex = .{},
    cond: host_io.Condition = .{},
    /// request id → channel. Entries inserted by `submit`, removed by `finish`.
    registry: std.AutoHashMapUnmanaged(u64, *ReqChannel) = .{},
    worker: ?std.Thread = null,
    stop: bool = false,

    /// Throughput counters (Effort 28, 3c — the aggregate-throughput gate). Only
    /// the worker thread writes them; `/stats` handler threads read via atomic
    /// load. `decode_*` cover the BATCHED decode step (the amortization win):
    /// `decode_tokens` sums the batch occupancy `ndec` per step, so
    /// decode_tokens/decode_wall_ns = aggregate decode tok/s and
    /// decode_tokens/decode_steps = mean batch occupancy. `prefill_*` cover the
    /// (still per-token B=1) prefill so the gate can see if admit dominates.
    decode_steps: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    decode_tokens: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    decode_wall_ns: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    prefill_tokens: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    prefill_wall_ns: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    peak_batch: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

    /// Allocate slot-based per-sequence state + a scheduler with `nslots` concurrent
    /// slots, each `slot_ctx` tokens deep. The forward (`fwd`) must already be
    /// initialized; it may be EITHER a gemma or qwen forward (dispatched by `Forward`).
    pub fn init(
        allocator: std.mem.Allocator,
        fwd: Forward,
        nslots: u32,
        slot_ctx: u32,
        eos: u32,
        eot: ?u32,
    ) !ServeEngine {
        if (nslots == 0 or nslots > scheduler_max) return error.InvalidParallelism;
        try fwd.allocSlots(nslots, slot_ctx);
        errdefer fwd.freeSlots();
        const sched = try scheduler.Scheduler.init(allocator, nslots);
        const capability: MtpCapability = if (nslots != 1)
            .{ .unavailable_reason = "MTP currently requires exactly one request slot." }
        else switch (fwd) {
            .gemma => .{ .unavailable_reason = "This model architecture does not expose the supported NextN/MTP layer." },
            .qwen => |q| prepareMtp(q),
            .inkling => |model| if (comptime is_rocm) prepareMtp(model) else unreachable,
        };
        log.info("CUDA serve engine ready: {d} slots × {d} ctx, eos={d}, eot={?d}, mtp_supported={}, mtp_default={}", .{
            nslots, slot_ctx, eos, eot, capability.supports_mtp, capability.default_enabled,
        });
        return .{ .allocator = allocator, .fwd = fwd, .sched = sched, .eos = eos, .eot = eot, .slot_ctx = slot_ctx, .mtp_capability = capability };
    }

    fn prepareMtp(model: anytype) MtpCapability {
        if (model.mtpUnavailableReason()) |reason| return .{ .unavailable_reason = reason };
        const default_enabled = model.mtpDefaultEnabled();
        const previous = model.setMtpExecutionOverride(true);
        defer _ = model.setMtpExecutionOverride(previous);
        const prepared = model.mtpPrepare() catch |err| {
            log.warn("Draft setup failed ({s}); ordinary decode remains available", .{@errorName(err)});
            return .{ .unavailable_reason = "Draft preparation failed; ordinary decoding remains available." };
        };
        if (!prepared) return .{ .unavailable_reason = "Draft preparation did not produce a usable context." };
        return .{ .supports_mtp = true, .default_enabled = default_enabled, .unavailable_reason = "" };
    }

    pub fn mtpCapability(self: *const ServeEngine) MtpCapability {
        return self.mtp_capability;
    }

    pub fn deinit(self: *ServeEngine) void {
        self.registry.deinit(self.allocator);
        self.sched.deinit();
        self.fwd.freeSlots();
    }

    /// Spawn the GPU worker thread.
    pub fn start(self: *ServeEngine) !void {
        self.worker = try std.Thread.spawn(.{}, workerLoop, .{self});
    }

    /// Signal the worker to drain outstanding work and exit, then join it.
    pub fn shutdown(self: *ServeEngine) void {
        self.mutex.lock();
        self.stop = true;
        self.mutex.unlock();
        self.cond.broadcast();
        if (self.worker) |w| w.join();
        self.worker = null;
    }

    /// Submit a request: register its channel, enqueue the prompt, wake the worker.
    /// The caller MUST keep `prompt_tokens` alive until the request finishes
    /// (the `Request` borrows the slice), and call `finish(id)` afterward.
    /// @returns the request id (used to wait/finish).
    pub fn submit(self: *ServeEngine, prompt_tokens: []const u32, max_tokens: u32, chan: *ReqChannel) !u64 {
        return self.submitWithOptions(prompt_tokens, max_tokens, chan, .{});
    }

    pub fn submitWithOptions(self: *ServeEngine, prompt_tokens: []const u32, max_tokens: u32, chan: *ReqChannel, options: SubmitOptions) !u64 {
        const enabled = try mtp_policy.resolveRequest(self.mtp_capability, options.enable_mtp);
        if (prompt_tokens.len == 0) return error.EmptyPrompt;
        if (prompt_tokens.len >= self.slot_ctx) return error.ContextTooSmall;
        if (max_tokens == 0) return error.InvalidMaxTokens;
        const token_budget = @min(max_tokens, self.slot_ctx - @as(u32, @intCast(prompt_tokens.len)));
        self.mutex.lock();
        defer self.mutex.unlock();
        // Reserve registry space before enqueueing, so an allocation failure
        // cannot leave a queued request borrowing a handler's destroyed channel.
        try self.registry.ensureUnusedCapacity(self.allocator, 1);
        const id = try self.sched.enqueue(prompt_tokens, .{ .max_tokens = token_budget });
        chan.mtp = .{ .requested = options.enable_mtp, .enabled = enabled };
        self.registry.putAssumeCapacity(id, chan);
        self.cond.broadcast();
        return id;
    }

    pub fn cancel(self: *ServeEngine, chan: *ReqChannel) void {
        self.mutex.lock();
        chan.cancelled = true;
        self.mutex.unlock();
        self.cond.broadcast();
    }

    /// Block until fresh tokens are available for `chan` or it finishes, then copy
    /// up to `dst.len` newly-generated tokens into `dst` (all under the lock, since
    /// the worker may realloc `chan.tokens` concurrently). Returns the count copied
    /// and whether the stream is now fully drained. Loop calling this until
    /// `finished`.
    pub fn nextChunk(self: *ServeEngine, chan: *ReqChannel, dst: []u32) Chunk {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (chan.tokens.items.len == chan.consumed and !chan.done and !chan.failed) {
            self.cond.wait(&self.mutex);
        }
        const avail = chan.tokens.items.len - chan.consumed;
        const n = @min(avail, dst.len);
        @memcpy(dst[0..n], chan.tokens.items[chan.consumed..][0..n]);
        chan.consumed += n;
        const drained = chan.consumed == chan.tokens.items.len;
        return .{
            .n = n,
            .finished = drained and (chan.done or chan.failed),
            .failed = chan.failed,
        };
    }

    /// Deregister + free a finished request's channel. Call once `done`/`failed`
    /// has been observed and all tokens drained.
    pub fn finish(self: *ServeEngine, id: u64, chan: *ReqChannel) void {
        self.mutex.lock();
        _ = self.registry.remove(id);
        self.mutex.unlock();
        chan.tokens.deinit(self.allocator);
    }

    pub fn executionFailed(self: *const ServeEngine) bool {
        self.fwd.checkExecution() catch return true;
        return false;
    }

    /// Write the cumulative throughput counters as JSON into `buf`. Lock-free
    /// (atomic loads) so `/stats` never contends the worker. The gate diffs two
    /// snapshots around each B-concurrent phase → pure decode tok/s + occupancy.
    pub fn statsJson(self: *ServeEngine, buf: []u8) ![]const u8 {
        return std.fmt.bufPrint(buf, "{{\"decode_steps\":{d},\"decode_tokens\":{d},\"decode_wall_ns\":{d}," ++
            "\"prefill_tokens\":{d},\"prefill_wall_ns\":{d},\"peak_batch\":{d}}}", .{
            self.decode_steps.load(.monotonic),
            self.decode_tokens.load(.monotonic),
            self.decode_wall_ns.load(.monotonic),
            self.prefill_tokens.load(.monotonic),
            self.prefill_wall_ns.load(.monotonic),
            self.peak_batch.load(.monotonic),
        });
    }

    // ── worker-thread internals ──────────────────────────────────────────────

    /// Append one token to a request's channel + wake its waiter (under lock).
    fn publish(self: *ServeEngine, id: u64, tok: u32) void {
        self.mutex.lock();
        if (self.registry.get(id)) |chan| {
            chan.tokens.append(self.allocator, tok) catch {
                chan.failed = true;
            };
        }
        self.mutex.unlock();
        self.cond.broadcast();
    }

    fn recordPrefillTiming(self: *ServeEngine, id: u64, tokens: usize, wall_ns: u64) void {
        self.mutex.lock();
        if (self.registry.get(id)) |chan| {
            chan.prefill_tokens += @intCast(tokens);
            chan.prefill_wall_ns += wall_ns;
        }
        self.mutex.unlock();
    }

    fn recordDecodeTiming(self: *ServeEngine, id: u64, wall_ns: u64) void {
        self.recordDecodeTimingCount(id, 1, wall_ns);
    }

    fn recordDecodeTimingCount(self: *ServeEngine, id: u64, tokens: u64, wall_ns: u64) void {
        self.mutex.lock();
        if (self.registry.get(id)) |chan| {
            chan.decode_tokens += tokens;
            chan.decode_wall_ns += wall_ns;
        }
        self.mutex.unlock();
    }

    fn requestMtpResult(self: *ServeEngine, id: u64) MtpResult {
        self.mutex.lock();
        defer self.mutex.unlock();
        return if (self.registry.get(id)) |chan| chan.mtp else .{};
    }

    fn recordMtpResult(self: *ServeEngine, id: u64, result: MtpResult) void {
        self.mutex.lock();
        if (self.registry.get(id)) |chan| chan.mtp = result;
        self.mutex.unlock();
    }

    fn requestCancelled(self: *ServeEngine, id: u64) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        return if (self.registry.get(id)) |chan| chan.cancelled else true;
    }

    fn publishChecked(self: *ServeEngine, id: u64, token: u32) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const chan = self.registry.get(id) orelse return error.RequestGone;
        try chan.tokens.append(self.allocator, token);
        self.cond.broadcast();
    }

    /// Mark a request finished (worker-only): flip its channel `done`, free the slot.
    fn finishSlot(self: *ServeEngine, slot_id: u32) void {
        const req = &self.sched.slots[slot_id].?;
        const id = req.id;
        req.transition(.completed) catch {};
        self.mutex.lock();
        if (self.registry.get(id)) |chan| chan.done = true;
        self.mutex.unlock();
        self.cond.broadcast();
        self.sched.release(slot_id); // worker-only; frees the slot for a waiter
    }

    /// Mark a request failed (worker-only) and free its slot.
    fn failSlot(self: *ServeEngine, slot_id: u32) void {
        const req = &self.sched.slots[slot_id].?;
        const id = req.id;
        req.transition(.failed) catch {};
        self.mutex.lock();
        if (self.registry.get(id)) |chan| chan.failed = true;
        self.mutex.unlock();
        self.cond.broadcast();
        self.sched.release(slot_id);
    }

    fn requestShouldStop(self: *ServeEngine, req: anytype) bool {
        if (self.requestCancelled(req.id)) return true;
        if (req.shouldStop(self.eos)) return true;
        const eot = self.eot orelse return false;
        if (req.generated_tokens.items.len == 0) return false;
        return req.generated_tokens.items[req.generated_tokens.items.len - 1] == eot;
    }

    /// MTP owns the single-sequence KV while ordinary requests use slot KV.
    /// Reset and prefill the corresponding state at every request boundary.
    /// Every metadata update completes before finishSlot/failSlot wakes a reader.
    fn runMtpRequest(self: *ServeEngine, slot_id: u32, timer_opt: *?host_io.Timer) bool {
        return switch (self.fwd) {
            .qwen => |q| self.runMtpWith(q, slot_id, timer_opt),
            .inkling => |model| if (comptime is_rocm) self.runMtpWith(model, slot_id, timer_opt) else unreachable,
            .gemma => false,
        };
    }

    fn runMtpWith(self: *ServeEngine, q: anytype, slot_id: u32, timer_opt: *?host_io.Timer) bool {
        const req = &self.sched.slots[slot_id].?;
        const id = req.id;
        var result = self.requestMtpResult(id);
        if (!result.enabled) return false;
        const previous = q.setMtpExecutionOverride(true);
        defer _ = q.setMtpExecutionOverride(previous);
        q.resetState() catch |err| {
            log.err("request {d} MTP state reset failed: {s}", .{ id, @errorName(err) });
            result.fallback_reason = "MTP state reset failed.";
            self.recordMtpResult(id, result);
            self.failSlot(slot_id);
            return true;
        };
        const pf_t0 = if (timer_opt.*) |*tm| tm.read() else 0;
        const seed = q.prefillBatched(req.prompt_tokens) catch |err| {
            self.fwd.checkExecution() catch {
                result.fallback_reason = "GPU execution failed during MTP prefill.";
                self.recordMtpResult(id, result);
                self.failSlot(slot_id);
                return true;
            };
            log.warn("request {d} MTP prefill failed ({s}); using ordinary slot decoding", .{ id, @errorName(err) });
            result.fallback_reason = "MTP prefill failed; used ordinary decoding.";
            self.recordMtpResult(id, result);
            return false;
        };
        const primed = q.mtpPrime(req.prompt_tokens) catch |err| blk: {
            self.fwd.checkExecution() catch {
                result.fallback_reason = "GPU execution failed during MTP priming.";
                self.recordMtpResult(id, result);
                self.failSlot(slot_id);
                return true;
            };
            log.warn("request {d} MTP priming failed ({s}); using ordinary slot decoding", .{ id, @errorName(err) });
            break :blk false;
        };
        if (!primed) {
            result.fallback_reason = "MTP priming failed; used ordinary decoding.";
            self.recordMtpResult(id, result);
            return false;
        }
        if (timer_opt.*) |*tm| {
            const elapsed = tm.read() - pf_t0;
            _ = self.prefill_wall_ns.fetchAdd(elapsed, .monotonic);
            const computed_tokens = self.fwd.computedPrefillTokens(req.prompt_tokens.len);
            _ = self.prefill_tokens.fetchAdd(computed_tokens, .monotonic);
            self.recordPrefillTiming(id, computed_tokens, elapsed);
        }
        req.transition(.decoding) catch {};
        var context = MtpDecodeContext(@TypeOf(q.*)){ .engine = self, .q = q, .slot_id = slot_id, .timer = timer_opt };
        mtp_policy.runDecode(&context, .{
            .max_tokens = req.params.max_tokens,
            .initial_position = @intCast(req.prompt_tokens.len),
            .max_context = @min(self.slot_ctx, q.max_ctx),
            .eos = self.eos,
            .eot = self.eot,
            .draft_limit = if (comptime @hasDecl(@TypeOf(q.*), "draftLimit")) q.draftLimit() else 3,
        }, seed, &result) catch |err| {
            log.err("request {d} MTP decoding failed: {s}", .{ id, @errorName(err) });
            result.fallback_reason = "MTP decoding failed after generation began.";
            self.recordMtpResult(id, result);
            self.failSlot(slot_id);
            return true;
        };
        self.recordMtpResult(id, result);
        if (result.cycles > 0) logMtpAcceptance(result.drafted, result.accepted, result.cycles);
        self.finishSlot(slot_id);
        return true;
    }

    fn MtpDecodeContext(comptime ForwardModel: type) type {
        const CycleResult = @typeInfo(@TypeOf(ForwardModel.mtpCycle)).@"fn".return_type.?;
        return struct {
            engine: *ServeEngine,
            q: *ForwardModel,
            slot_id: u32,
            timer: *?host_io.Timer,

            pub fn isCancelled(self: *@This()) bool {
                return self.engine.requestCancelled(self.engine.sched.slots[self.slot_id].?.id);
            }

            pub fn cycle(self: *@This(), seed: u32, pos: u32, budget: u32, eos: u32) CycleResult {
                const cycle_start = if (self.timer.*) |*tm| tm.read() else 0;
                const result = try self.q.mtpCycle(seed, pos, budget, eos);
                _ = self.engine.decode_steps.fetchAdd(1, .monotonic);
                if (self.timer.*) |*tm| {
                    const elapsed = tm.read() - cycle_start;
                    _ = self.engine.decode_wall_ns.fetchAdd(elapsed, .monotonic);
                    self.engine.recordDecodeTimingCount(self.engine.sched.slots[self.slot_id].?.id, 0, elapsed);
                }
                if (self.engine.peak_batch.load(.monotonic) < 1) self.engine.peak_batch.store(1, .monotonic);
                return result;
            }

            pub fn emit(self: *@This(), token: u32, is_decode: bool) !void {
                const req = &self.engine.sched.slots[self.slot_id].?;
                try req.appendToken(token);
                try self.engine.publishChecked(req.id, token);
                if (is_decode) {
                    _ = self.engine.decode_tokens.fetchAdd(1, .monotonic);
                    self.engine.recordDecodeTimingCount(req.id, 1, 0);
                }
            }
        };
    }

    fn workerLoop(self: *ServeEngine) void {
        const fwd = self.fwd;
        // Local copies of the scratch slot-id slices (scratch is reused across
        // pendingPrefill/activeDecoding; copying decouples iteration from it).
        var prefill_buf: [scheduler_max]u32 = undefined;
        var dec_buf: [scheduler_max]u32 = undefined;
        var tks: [scheduler_max]u32 = undefined;
        var pss: [scheduler_max]u32 = undefined;
        var sls: [scheduler_max]u32 = undefined;
        var out: [scheduler_max]u32 = undefined;
        // Monotonic timer for the 3c throughput counters; non-fatal if absent.
        var timer_opt: ?host_io.Timer = host_io.Timer.start() catch null;

        while (true) {
            // Wait for work (or shutdown) while idle — event-driven, no poll.
            self.mutex.lock();
            while (!self.stop and self.sched.isIdle()) self.cond.wait(&self.mutex);
            if (self.stop and self.sched.isIdle()) {
                self.mutex.unlock();
                break;
            }
            // Admit every pending arrival that fits a free slot.
            while ((self.sched.admitNext() catch null) != null) {}
            self.mutex.unlock();

            // PREFILL each freshly-admitted slot (per-token B=1 decodeBatch into
            // its slot), publish the first generated token, promote to .decoding.
            // `pendingPrefill` aliases sched.scratch → snapshot it first.
            const pf = self.sched.pendingPrefill();
            const npf = pf.len;
            @memcpy(prefill_buf[0..npf], pf);
            for (prefill_buf[0..npf]) |slot_id| {
                const req = &self.sched.slots[slot_id].?;
                const np = req.prompt_tokens.len;
                if (self.requestCancelled(req.id)) {
                    self.finishSlot(slot_id);
                    continue;
                }
                if (self.runMtpRequest(slot_id, &timer_opt)) continue;
                const pf_t0 = if (timer_opt) |*tm| tm.read() else 0;
                // Clear any accumulated state from this slot's previous request
                // BEFORE prefilling the new one from pos=0 (qwen SSM recurrent state;
                // no-op for gemma). Slots are reused when nslots < concurrent clients.
                fwd.resetSlot(slot_id) catch |err| {
                    log.err("request {d} slot reset failed: {s}", .{ req.id, @errorName(err) });
                    self.failSlot(slot_id);
                    continue;
                };
                // Short single-slot MoE uses the same batched prefill and SSM
                // reduction as CLI. Other short prompts keep per-token prefill;
                // long prompts retain the existing batched path.
                // A/B toggle: ZINC_BATCHED_PREFILL=0 disables (uses per-token fallback).
                const cli_moe_short = switch (fwd) {
                    .qwen => |q| q.useCliMoeSlotPrefill(np),
                    .gemma, .inkling => false,
                };
                const use_batched = (np >= prefill_policy.batch_threshold or cli_moe_short or fwd == .inkling) and
                    (host_io.getenv("ZINC_BATCHED_PREFILL") == null or
                        !std.mem.eql(u8, host_io.getenv("ZINC_BATCHED_PREFILL").?, "0"));
                const batched_tok = if (use_batched)
                    (fwd.prefillSlot(req.prompt_tokens, slot_id) catch |err| blk: {
                        log.warn("request {d} batched slot prefill failed ({s}); resetting before fallback", .{ req.id, @errorName(err) });
                        // A failed chunk/layer may already have advanced the
                        // recurrent state. Restart the fallback at position 0
                        // only after clearing that partial state. A poisoned
                        // GPU context fails reset and never enters fallback.
                        fwd.resetSlot(slot_id) catch |reset_err| {
                            log.err("request {d} fallback reset failed: {s}", .{ req.id, @errorName(reset_err) });
                            self.failSlot(slot_id);
                            continue;
                        };
                        break :blk null;
                    })
                else
                    null;
                if (batched_tok) |tok| {
                    const computed_tokens = fwd.computedPrefillTokens(np);
                    if (timer_opt) |*tm| {
                        const elapsed = tm.read() - pf_t0;
                        _ = self.prefill_wall_ns.fetchAdd(elapsed, .monotonic);
                        _ = self.prefill_tokens.fetchAdd(computed_tokens, .monotonic);
                        self.recordPrefillTiming(req.id, computed_tokens, elapsed);
                    }
                    req.appendToken(tok) catch {
                        self.failSlot(slot_id);
                        continue;
                    };
                    req.transition(.decoding) catch {};
                    self.publish(req.id, tok);
                    if (self.requestShouldStop(req)) self.finishSlot(slot_id);
                } else {
                    // Fallback: per-token B=1 decodeBatch prefill (gemma or error).
                    var pos: u32 = 0;
                    var tok: u32 = 0;
                    var k: usize = 0;
                    while (k < np) : (k += 1) {
                        var tk = [_]u32{req.prompt_tokens[k]};
                        var ps = [_]u32{pos};
                        var sl = [_]u32{slot_id};
                        var ot = [_]u32{0};
                        fwd.decodeBatch(&tk, &ps, &sl, &ot) catch |err| {
                            log.err("request {d} prefill failed at token {d}/{d}: {s}", .{ req.id, k + 1, np, @errorName(err) });
                            self.failSlot(slot_id);
                            break;
                        };
                        tok = ot[0];
                        pos += 1;
                    } else {
                        if (timer_opt) |*tm| {
                            const elapsed = tm.read() - pf_t0;
                            _ = self.prefill_wall_ns.fetchAdd(elapsed, .monotonic);
                            _ = self.prefill_tokens.fetchAdd(np, .monotonic);
                            self.recordPrefillTiming(req.id, np, elapsed);
                        }
                        req.appendToken(tok) catch {
                            self.failSlot(slot_id);
                            continue;
                        };
                        req.transition(.decoding) catch {};
                        self.publish(req.id, tok);
                        if (self.requestShouldStop(req)) self.finishSlot(slot_id);
                    }
                }
            }

            // DECODE one batched step over the whole running batch (mixed
            // positions/slots) — the amortization win: weights read once for all B.
            const dec = self.sched.activeDecoding();
            @memcpy(dec_buf[0..dec.len], dec);
            var ndec: usize = 0;
            for (dec_buf[0..dec.len]) |slot_id| {
                const req = &self.sched.slots[slot_id].?;
                if (self.requestCancelled(req.id)) {
                    self.finishSlot(slot_id);
                    continue;
                }
                const gen_n = req.generated_tokens.items.len;
                tks[ndec] = req.generated_tokens.items[gen_n - 1];
                pss[ndec] = @intCast(req.prompt_tokens.len + gen_n - 1);
                sls[ndec] = slot_id;
                dec_buf[ndec] = slot_id;
                ndec += 1;
            }
            if (ndec > 0) {
                const dec_t0 = if (timer_opt) |*tm| tm.read() else 0;
                fwd.decodeBatch(tks[0..ndec], pss[0..ndec], sls[0..ndec], out[0..ndec]) catch |err| {
                    log.err("batched decode failed for {d} active request(s): {s}", .{ ndec, @errorName(err) });
                    for (dec_buf[0..ndec]) |slot_id| {
                        if (self.sched.slots[slot_id] != null) self.failSlot(slot_id);
                    }
                    continue;
                };
                var dec_elapsed: u64 = 0;
                if (timer_opt) |*tm| {
                    dec_elapsed = tm.read() - dec_t0;
                    _ = self.decode_wall_ns.fetchAdd(dec_elapsed, .monotonic);
                    _ = self.decode_tokens.fetchAdd(ndec, .monotonic);
                    _ = self.decode_steps.fetchAdd(1, .monotonic);
                    if (ndec > self.peak_batch.load(.monotonic))
                        self.peak_batch.store(ndec, .monotonic);
                }
                for (dec_buf[0..ndec], 0..) |slot_id, i| {
                    const req = &self.sched.slots[slot_id].?;
                    if (dec_elapsed != 0) self.recordDecodeTiming(req.id, dec_elapsed);
                    req.appendToken(out[i]) catch {
                        self.failSlot(slot_id);
                        continue;
                    };
                    self.publish(req.id, out[i]);
                    if (self.requestShouldStop(req)) self.finishSlot(slot_id);
                }
            }
        }
        log.info("CUDA serve worker exited", .{});
    }
};

/// Upper bound on concurrent slots the worker's stack buffers can hold. The
/// scheduler enforces the real limit (`max_parallel`); this just sizes the
/// fixed worker scratch. 64 is far above any practical single-GPU batch.
const scheduler_max = 64;
