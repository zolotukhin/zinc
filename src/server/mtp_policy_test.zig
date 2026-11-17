const std = @import("std");
const mtp = @import("mtp_policy.zig");

test "MTP requests preserve explicit overrides and engine defaults" {
    const default_on = mtp.Capability{ .supports_mtp = true, .default_enabled = true, .unavailable_reason = "" };
    try std.testing.expect(try mtp.resolveRequest(default_on, null));
    try std.testing.expect(try mtp.resolveRequest(default_on, true));
    try std.testing.expect(!try mtp.resolveRequest(default_on, false));

    const default_off = mtp.Capability{ .supports_mtp = true, .default_enabled = false, .unavailable_reason = "" };
    try std.testing.expect(!try mtp.resolveRequest(default_off, null));
    try std.testing.expect(try mtp.resolveRequest(default_off, true));
    try std.testing.expect(!try mtp.resolveRequest(default_off, false));
    // Resolving one request must never alter the default for the next request.
    try std.testing.expect(try mtp.resolveRequest(default_on, null));
    try std.testing.expect(!try mtp.resolveRequest(default_off, null));
}

test "MTP unavailable requests reject explicit opt-in but allow ordinary decoding" {
    const unsupported = mtp.Capability{ .supports_mtp = false, .default_enabled = false, .unavailable_reason = "Multiple slots are active" };
    try std.testing.expectError(error.MtpUnavailable, mtp.resolveRequest(unsupported, true));
    try std.testing.expect(!try mtp.resolveRequest(unsupported, false));
    try std.testing.expect(!try mtp.resolveRequest(unsupported, null));
}

test "MTP draft budgets reserve the target seed and obey output and context limits" {
    try std.testing.expectEqual(@as(u32, 0), mtp.draftBudget(0, 10, 100));
    try std.testing.expectEqual(@as(u32, 3), mtp.draftBudget(100, 10, 100));
    try std.testing.expectEqual(@as(u32, 2), mtp.draftBudget(2, 10, 100));
    try std.testing.expectEqual(@as(u32, 1), mtp.draftBudget(100, 98, 100));
    try std.testing.expectEqual(@as(u32, 0), mtp.draftBudget(100, 99, 100));
    try std.testing.expectEqual(@as(u32, 0), mtp.draftBudget(100, 100, 100));
    try std.testing.expectEqual(@as(u32, 0), mtp.draftBudget(100, 101, 100));
}

const Cycle = struct {
    drafts: [3]u32,
    n_drafted: u32,
    n_accepted: u32,
    next_token: u32,
};

const Call = struct {
    seed: u32,
    position: u32,
    budget: u32,
    eos: u32,
};

const FakeContext = struct {
    scripted_cycles: []const Cycle = &.{},
    calls: [8]Call = undefined,
    call_count: usize = 0,
    tokens: [16]u32 = @splat(0),
    is_decode: [16]bool = @splat(false),
    count: usize = 0,
    cancel_after: ?usize = null,
    fail_cycle: ?usize = null,
    fail_emit: ?usize = null,

    pub fn cycle(self: *FakeContext, seed: u32, position: u32, budget: u32, eos: u32) !Cycle {
        const index = self.call_count;
        self.calls[index] = .{ .seed = seed, .position = position, .budget = budget, .eos = eos };
        self.call_count += 1;
        if (self.fail_cycle == index) return error.TestCycleFailure;
        if (index >= self.scripted_cycles.len) return error.UnexpectedCycle;
        const result = self.scripted_cycles[index];
        if (result.n_drafted > budget) return error.DraftBudgetExceeded;
        return result;
    }

    pub fn emit(self: *FakeContext, token: u32, is_decode: bool) !void {
        if (self.fail_emit == self.count) return error.TestEmitFailure;
        self.tokens[self.count] = token;
        self.is_decode[self.count] = is_decode;
        self.count += 1;
    }

    pub fn isCancelled(self: *FakeContext) bool {
        return if (self.cancel_after) |limit| self.count >= limit else false;
    }

    fn emitted(self: *const FakeContext) []const u32 {
        return self.tokens[0..self.count];
    }
};

fn freshResult() mtp.Result {
    return .{ .requested = true, .enabled = true, .used = false, .drafted = 0, .accepted = 0, .cycles = 0, .fallback_reason = null };
}

fn limits(max_tokens: u32) mtp.DecodeLimits {
    return .{ .max_tokens = max_tokens, .initial_position = 20, .max_context = 100, .eos = 999 };
}

test "MTP decode publishes accepted drafts and corrected target token without rejected suffix" {
    var ctx = FakeContext{ .scripted_cycles = &.{.{ .drafts = .{ 10, 998, 997 }, .n_drafted = 2, .n_accepted = 1, .next_token = 20 }} };
    var stats = freshResult();
    try mtp.runDecode(&ctx, limits(3), 5, &stats);
    try std.testing.expectEqualSlices(u32, &.{ 5, 10, 20 }, ctx.emitted());
    try std.testing.expect(!ctx.is_decode[0]);
    try std.testing.expect(ctx.is_decode[1] and ctx.is_decode[2]);
    try std.testing.expectEqual(@as(u64, 1), stats.cycles);
    try std.testing.expectEqual(@as(u64, 2), stats.drafted);
    try std.testing.expectEqual(@as(u64, 1), stats.accepted);
    try std.testing.expect(stats.used);
}

test "MTP decode with zero accepted drafts emits only the corrected target token" {
    var ctx = FakeContext{ .scripted_cycles = &.{.{ .drafts = .{ 998, 0, 0 }, .n_drafted = 1, .n_accepted = 0, .next_token = 20 }} };
    var stats = freshResult();
    try mtp.runDecode(&ctx, limits(2), 5, &stats);
    try std.testing.expectEqualSlices(u32, &.{ 5, 20 }, ctx.emitted());
    try std.testing.expectEqual(@as(u64, 0), stats.accepted);
    try std.testing.expectEqual(@as(u64, 1), stats.cycles);
    try std.testing.expectEqual(@as(u32, 1), ctx.calls[0].budget);
}

test "MTP decode never cycles for one-token output and clamps accepted prefix to final budget" {
    var one = FakeContext{};
    var one_stats = freshResult();
    try mtp.runDecode(&one, limits(1), 5, &one_stats);
    try std.testing.expectEqualSlices(u32, &.{5}, one.emitted());
    try std.testing.expectEqual(@as(usize, 0), one.call_count);
    try std.testing.expect(!one_stats.used);

    var two = FakeContext{ .scripted_cycles = &.{.{ .drafts = .{ 10, 0, 0 }, .n_drafted = 1, .n_accepted = 1, .next_token = 20 }} };
    var two_stats = freshResult();
    try mtp.runDecode(&two, limits(2), 5, &two_stats);
    try std.testing.expectEqualSlices(u32, &.{ 5, 10 }, two.emitted());

    var four = FakeContext{ .scripted_cycles = &.{.{ .drafts = .{ 10, 11, 12 }, .n_drafted = 3, .n_accepted = 3, .next_token = 20 }} };
    var four_stats = freshResult();
    try mtp.runDecode(&four, limits(4), 5, &four_stats);
    try std.testing.expectEqualSlices(u32, &.{ 5, 10, 11, 12 }, four.emitted());
}

test "MTP decode never reads or emits beyond the remaining context" {
    var ctx = FakeContext{ .scripted_cycles = &.{.{ .drafts = .{ 10, 0, 0 }, .n_drafted = 1, .n_accepted = 1, .next_token = 20 }} };
    var stats = freshResult();
    try mtp.runDecode(&ctx, .{ .max_tokens = 10, .initial_position = 98, .max_context = 100, .eos = 999 }, 5, &stats);
    try std.testing.expectEqualSlices(u32, &.{ 5, 10 }, ctx.emitted());
    try std.testing.expectEqual(@as(u32, 1), ctx.calls[0].budget);

    var last = FakeContext{};
    var last_stats = freshResult();
    try mtp.runDecode(&last, .{ .max_tokens = 10, .initial_position = 99, .max_context = 100, .eos = 999 }, 5, &last_stats);
    try std.testing.expectEqualSlices(u32, &.{5}, last.emitted());
    try std.testing.expectEqual(@as(usize, 0), last.call_count);
}

test "MTP decode advances seed and position by target-committed tokens between cycles" {
    var ctx = FakeContext{ .scripted_cycles = &.{
        .{ .drafts = .{ 10, 998, 997 }, .n_drafted = 3, .n_accepted = 1, .next_token = 20 },
        .{ .drafts = .{ 30, 0, 0 }, .n_drafted = 1, .n_accepted = 1, .next_token = 40 },
    } };
    var stats = freshResult();
    try mtp.runDecode(&ctx, limits(4), 5, &stats);
    try std.testing.expectEqualSlices(u32, &.{ 5, 10, 20, 30 }, ctx.emitted());
    try std.testing.expectEqual(@as(u32, 5), ctx.calls[0].seed);
    try std.testing.expectEqual(@as(u32, 20), ctx.calls[0].position);
    try std.testing.expectEqual(@as(u32, 20), ctx.calls[1].seed);
    try std.testing.expectEqual(@as(u32, 22), ctx.calls[1].position);
    try std.testing.expectEqual(@as(u64, 2), stats.cycles);
    try std.testing.expectEqual(@as(u64, 2), stats.accepted);
}

test "MTP decode stops at EOS and EOT before later accepted drafts or target prediction" {
    const cycle = Cycle{ .drafts = .{ 10, 11, 12 }, .n_drafted = 3, .n_accepted = 3, .next_token = 20 };
    var eos = FakeContext{ .scripted_cycles = &.{cycle} };
    var eos_stats = freshResult();
    try mtp.runDecode(&eos, .{ .max_tokens = 10, .initial_position = 20, .max_context = 100, .eos = 11 }, 5, &eos_stats);
    try std.testing.expectEqualSlices(u32, &.{ 5, 10, 11 }, eos.emitted());
    var eot = FakeContext{ .scripted_cycles = &.{cycle} };
    var eot_stats = freshResult();
    try mtp.runDecode(&eot, .{ .max_tokens = 10, .initial_position = 20, .max_context = 100, .eos = 999, .eot = 10 }, 5, &eot_stats);
    try std.testing.expectEqualSlices(u32, &.{ 5, 10 }, eot.emitted());

    var seed_eos = FakeContext{};
    var seed_stats = freshResult();
    try mtp.runDecode(&seed_eos, .{ .max_tokens = 10, .initial_position = 20, .max_context = 100, .eos = 5 }, 5, &seed_stats);
    try std.testing.expectEqualSlices(u32, &.{5}, seed_eos.emitted());
    try std.testing.expectEqual(@as(usize, 0), seed_eos.call_count);
}

test "MTP decode observes cancellation between accepted draft emissions" {
    var ctx = FakeContext{ .scripted_cycles = &.{.{ .drafts = .{ 10, 11, 12 }, .n_drafted = 3, .n_accepted = 3, .next_token = 20 }}, .cancel_after = 2 };
    var stats = freshResult();
    try mtp.runDecode(&ctx, limits(10), 5, &stats);
    try std.testing.expectEqualSlices(u32, &.{ 5, 10 }, ctx.emitted());
    var cancelled = FakeContext{ .cancel_after = 0 };
    var cancelled_stats = freshResult();
    try mtp.runDecode(&cancelled, limits(10), 5, &cancelled_stats);
    try std.testing.expectEqual(@as(usize, 0), cancelled.count);
    try std.testing.expectEqual(@as(usize, 0), cancelled.call_count);
}

test "MTP decode errors preserve already-published output and avoid phantom successful cycles" {
    var ctx = FakeContext{ .fail_cycle = 0 };
    var stats = freshResult();
    try std.testing.expectError(error.TestCycleFailure, mtp.runDecode(&ctx, limits(10), 5, &stats));
    try std.testing.expectEqualSlices(u32, &.{5}, ctx.emitted());
    try std.testing.expectEqual(@as(u64, 0), stats.cycles);
    try std.testing.expect(!stats.used);
}
