//! @section API Server
//! MTP request policy and verified token publication.
//!
//! Resolves each request's override against immutable runtime capabilities and
//! bounds speculative output by token, context, cancellation, and stop limits.
//! The same publication driver runs with GPU execution or scripted CPU tests.
const std = @import("std");

pub const Capability = struct {
    supports_mtp: bool = false,
    default_enabled: bool = false,
    unavailable_reason: []const u8 = "MTP is unavailable in this runtime.",
};

pub const SubmitOptions = struct {
    enable_mtp: ?bool = null,
};

pub const Result = struct {
    /// Original request preference; null preserves the server default.
    requested: ?bool = null,
    /// Effective, latched preference for this request.
    enabled: bool = false,
    /// True only after at least one actual draft/verification cycle succeeds.
    used: bool = false,
    drafted: u64 = 0,
    /// Drafts accepted by the verifier (some may follow a stop token/budget).
    accepted: u64 = 0,
    cycles: u64 = 0,
    fallback_reason: ?[]const u8 = null,
};

pub fn resolveRequest(capability: Capability, requested: ?bool) !bool {
    if (requested == true and !capability.supports_mtp) return error.MtpUnavailable;
    return capability.supports_mtp and (requested orelse capability.default_enabled);
}

pub fn draftBudget(remaining_output: u32, position: u32, max_context: u32) u32 {
    return draftBudgetForLimit(3, remaining_output, position, max_context);
}

fn draftBudgetForLimit(limit: u32, remaining_output: u32, position: u32, max_context: u32) u32 {
    // Reserve a context position for the seed that was already published.
    return @min(limit, @min(remaining_output, (max_context -| position) -| 1));
}

pub const DecodeLimits = struct {
    max_tokens: u32,
    initial_position: u32,
    max_context: u32,
    eos: u32,
    eot: ?u32 = null,
    draft_limit: u32 = 3,
};

fn stops(token: u32, limits: DecodeLimits) bool {
    return token == limits.eos or (limits.eot != null and token == limits.eot.?);
}

/// Publish only target-verified output. Context owns GPU execution and publishing,
/// but must not mark the request complete: the caller first snapshots `stats`
/// into its result channel, then signals completion/release.
///
/// Context methods: cycle(seed,pos,budget,eos)!Cycle; emit(token,is_decode)!void;
/// isCancelled() bool. The first seed is the prefill output; all later emitted
/// tokens count as decode output, even when a verified prefix hits a stop early.
pub fn runDecode(ctx: anytype, limits: DecodeLimits, seed: u32, stats: *Result) !void {
    if (limits.initial_position >= limits.max_context) return error.ContextTooSmall;
    const max_tokens = @min(limits.max_tokens, limits.max_context - limits.initial_position);
    if (max_tokens == 0 or ctx.isCancelled()) return;

    var next = seed;
    var position = limits.initial_position;
    var produced: u32 = 1;
    try ctx.emit(next, false);
    if (stops(next, limits)) return;

    while (produced < max_tokens) {
        if (ctx.isCancelled()) return;
        const budget = draftBudgetForLimit(limits.draft_limit, max_tokens - produced, position, limits.max_context);
        if (budget == 0) return;
        const result = try ctx.cycle(next, position, budget, limits.eos);
        stats.used = true;
        stats.cycles += 1;
        stats.drafted += result.n_drafted;
        stats.accepted += result.n_accepted;
        position += 1 + result.n_accepted;

        var i: u32 = 0;
        while (i < result.n_accepted) : (i += 1) {
            if (ctx.isCancelled()) return;
            const token = result.drafts[i];
            try ctx.emit(token, true);
            produced += 1;
            if (produced >= max_tokens or stops(token, limits)) return;
        }
        if (ctx.isCancelled()) return;
        next = result.next_token;
        try ctx.emit(next, true);
        produced += 1;
        if (stops(next, limits)) return;
    }
}
