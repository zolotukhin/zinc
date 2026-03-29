//! Request lifecycle management for concurrent inference serving.
//! @section Scheduler
//! Each incoming API request maps to a Request that tracks its state
//! through prefill, decode, and completion phases.
const std = @import("std");

const log = std.log.scoped(.request);

/// Request processing state machine.
/// Valid transitions: pending → prefilling → decoding → completed,
/// with cancelled or failed reachable from any active state.
pub const RequestState = enum {
    /// Queued, waiting for a prefill slot.
    pending,
    /// Prompt tokens are being processed.
    prefilling,
    /// Output tokens are being generated.
    decoding,
    /// Finished normally (EOS or max_tokens reached).
    completed,
    /// Client disconnected before completion.
    cancelled,
    /// Generation terminated due to a runtime error.
    failed,
};

/// Generation parameters from the API request.
pub const GenerationParams = struct {
    /// Max tokens to generate.
    max_tokens: u32 = 256,
    /// Sampling temperature.
    temperature: f32 = 1.0,
    /// Nucleus sampling threshold.
    top_p: f32 = 1.0,
    /// Top-k candidates.
    top_k: u32 = 50,
    /// Enable SSE streaming.
    stream: bool = true,
    /// Custom stop sequences.
    stop_sequences: []const []const u8 = &.{},
};

/// A single inference request with its lifecycle state.
pub const Request = struct {
    /// Unique identifier.
    id: u64,
    /// Current lifecycle state.
    state: RequestState,
    /// Tokenized prompt.
    prompt_tokens: []const u32,
    /// Generated token IDs.
    generated_tokens: std.ArrayList(u32),
    /// Generation parameters.
    params: GenerationParams,
    // KV cache slot assignment (set by scheduler)
    /// KV cache slot, or null.
    slot_id: ?u32,
    // Timing
    /// Creation timestamp.
    created_at_ns: i128,
    /// First token timestamp.
    first_token_ns: ?i128,
    /// Allocator for owned resources.
    allocator: std.mem.Allocator,

    /// Create a new request in the pending state with the given prompt and parameters.
    /// @param allocator Allocator for the generated token buffer.
    /// @param id Unique request identifier.
    /// @param prompt_tokens Tokenized prompt (owned by the caller).
    /// @param params Generation parameters (max_tokens, temperature, etc.).
    /// @returns A Request ready to be submitted to the scheduler.
    pub fn init(allocator: std.mem.Allocator, id: u64, prompt_tokens: []const u32, params: GenerationParams) Request {
        return .{
            .id = id,
            .state = .pending,
            .prompt_tokens = prompt_tokens,
            .generated_tokens = .{},
            .params = params,
            .slot_id = null,
            .created_at_ns = std.time.nanoTimestamp(),
            .first_token_ns = null,
            .allocator = allocator,
        };
    }

    /// Transition to the next lifecycle state.
    /// @param self Request to transition.
    /// @param new_state Target state (must be a valid successor of the current state).
    /// @note Returns error.InvalidTransition if the state change is not allowed.
    pub fn transition(self: *Request, new_state: RequestState) !void {
        const valid = switch (self.state) {
            .pending => new_state == .prefilling or new_state == .cancelled,
            .prefilling => new_state == .decoding or new_state == .failed or new_state == .cancelled,
            .decoding => new_state == .completed or new_state == .failed or new_state == .cancelled,
            .completed, .cancelled, .failed => false,
        };
        if (!valid) return error.InvalidTransition;
        self.state = new_state;
    }

    /// Append a generated token and record the first-token timestamp if unset.
    /// @param self Request to append to.
    /// @param token Generated token ID to add.
    pub fn appendToken(self: *Request, token: u32) !void {
        if (self.first_token_ns == null) {
            self.first_token_ns = std.time.nanoTimestamp();
        }
        try self.generated_tokens.append(self.allocator, token);
    }

    /// Check if generation should stop (max_tokens or EOS).
    pub fn shouldStop(self: *const Request, eos_token_id: u32) bool {
        if (self.generated_tokens.items.len >= self.params.max_tokens) return true;
        if (self.generated_tokens.items.len > 0) {
            const last = self.generated_tokens.items[self.generated_tokens.items.len - 1];
            if (last == eos_token_id) return true;
        }
        return false;
    }

    /// Release the generated token buffer owned by this request.
    /// Release the generated token buffer owned by this request.
    pub fn deinit(self: *Request) void {
        self.generated_tokens.deinit(self.allocator);
    }
};

test "Request state transitions" {
    const allocator = std.testing.allocator;
    var req = Request.init(allocator, 1, &.{}, .{});
    defer req.deinit();

    try std.testing.expectEqual(RequestState.pending, req.state);
    try req.transition(.prefilling);
    try std.testing.expectEqual(RequestState.prefilling, req.state);
    try req.transition(.decoding);
    try std.testing.expectEqual(RequestState.decoding, req.state);
    try req.transition(.completed);
    try std.testing.expectEqual(RequestState.completed, req.state);
}

test "Request stops at max_tokens" {
    const allocator = std.testing.allocator;
    var req = Request.init(allocator, 1, &.{}, .{ .max_tokens = 3 });
    defer req.deinit();
    try req.appendToken(100);
    try req.appendToken(200);
    try std.testing.expect(!req.shouldStop(999));
    try req.appendToken(300);
    try std.testing.expect(req.shouldStop(999));
}

test "Request stops at EOS token" {
    const allocator = std.testing.allocator;
    var req = Request.init(allocator, 1, &.{}, .{ .max_tokens = 100 });
    defer req.deinit();
    try req.appendToken(10);
    try std.testing.expect(!req.shouldStop(42));
    try req.appendToken(42); // EOS
    try std.testing.expect(req.shouldStop(42));
}

test "Request appendToken records first token time" {
    const allocator = std.testing.allocator;
    var req = Request.init(allocator, 1, &.{}, .{});
    defer req.deinit();
    try std.testing.expect(req.first_token_ns == null);
    try req.appendToken(1);
    try std.testing.expect(req.first_token_ns != null);
    // Second append should not change first_token_ns
    const first = req.first_token_ns.?;
    try req.appendToken(2);
    try std.testing.expectEqual(first, req.first_token_ns.?);
}

test "Request invalid state transition fails" {
    const allocator = std.testing.allocator;
    var req = Request.init(allocator, 1, &.{}, .{});
    defer req.deinit();
    // pending → completed should fail (must go through prefilling/decoding)
    try std.testing.expectError(error.InvalidTransition, req.transition(.completed));
}

test "Request generation params defaults" {
    const params = GenerationParams{};
    try std.testing.expectEqual(@as(u32, 256), params.max_tokens);
    try std.testing.expectEqual(@as(f32, 1.0), params.temperature);
    try std.testing.expectEqual(@as(f32, 1.0), params.top_p);
    try std.testing.expect(params.stream);
}
