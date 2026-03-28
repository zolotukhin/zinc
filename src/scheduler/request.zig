//! Request lifecycle management for concurrent inference serving.
//! @section Scheduler
//! Each incoming API request maps to a Request that tracks its state
//! through prefill, decode, and completion phases.
const std = @import("std");

const log = std.log.scoped(.request);

/// Request processing state machine.
pub const RequestState = enum {
    pending, // queued, waiting for prefill
    prefilling, // prompt tokens being processed
    decoding, // generating output tokens
    completed, // finished (EOS or max_tokens)
    cancelled, // client disconnected
    failed, // runtime error
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

    /// Create a new request in pending state.
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

    /// Transition to the next state. Returns error if transition is invalid.
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

    /// Append a generated token. Records first-token time.
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
