//! Automatic chat session reuse policy.
//! @section API Server
//! Limits capability discovery to the backend and model state that have passed
//! cached/native-reference and tool-history controls. Explicit API session keys
//! retain their existing behavior independently of automatic client discovery.
const std = @import("std");

/// Attention-only Vulkan models have passed cached/native and tool-history
/// controls. The hybrid control returned correct full-prefill output, but its
/// stored prefix and recurrent checkpoint positions differed. Keep clients on
/// full prefill until recurrent reuse has its own successful qualification.
pub fn canAdvertise(vulkan: bool, recurrent_width: u32) bool {
    return vulkan and recurrent_width == 0;
}

/// KV rows before an edited token remain valid for attention-only models.
/// Recurrent state and speculative draft state cannot be rewound by changing
/// the token position, so they must retain the existing full-prefill fallback.
pub fn canRewindPrefix(vulkan: bool, recurrent_width: u32, has_draft_state: bool, has_kv_pages: bool) bool {
    return canAdvertise(vulkan, recurrent_width) and !has_draft_state and has_kv_pages;
}

/// The Intel Gemma MoE client path needs canonical replay after its template
/// drops an empty thought channel. Keep the automatic change on this profile;
/// other attention-only graphs retain the explicit experimental override.
pub fn defaultRewind(vulkan: bool, intel_xe2: bool, gemma: bool, experts: u32, recurrent_width: u32) bool {
    return vulkan and intel_xe2 and gemma and experts > 0 and recurrent_width == 0;
}

pub fn rewindRequested(setting: ?[]const u8, qualified_default: bool) bool {
    return if (setting) |value| std.mem.eql(u8, value, "1") else qualified_default;
}

/// Reuse only the identical canonical prefix. Reprocess at least the last
/// prompt token even on an identical/shortened retry: the previous request's
/// logits describe its final generated token, not the end of this prompt.
pub fn rewindPrefixLen(stored: []const u32, prompt: []const u32) usize {
    if (prompt.len == 0) return 0;
    var common: usize = 0;
    const limit = @min(stored.len, prompt.len - 1);
    while (common < limit and stored[common] == prompt[common]) : (common += 1) {}
    return common;
}

test "automatic reuse excludes recurrent state and unqualified backends" {
    try std.testing.expect(canAdvertise(true, 0));
    try std.testing.expect(!canAdvertise(true, 4096));
    try std.testing.expect(!canAdvertise(false, 0));
    try std.testing.expect(!canAdvertise(false, 4096));
}

test "prefix rewind requires resident attention KV without draft state" {
    try std.testing.expect(canRewindPrefix(true, 0, false, true));
    try std.testing.expect(!canRewindPrefix(false, 0, false, true));
    try std.testing.expect(!canRewindPrefix(true, 4096, false, true));
    try std.testing.expect(!canRewindPrefix(true, 0, true, true));
    try std.testing.expect(!canRewindPrefix(true, 0, false, false));
}

test "automatic canonical replay stays on Intel Gemma MoE and honors explicit controls" {
    try std.testing.expect(defaultRewind(true, true, true, 128, 0));
    try std.testing.expect(!defaultRewind(false, true, true, 128, 0));
    try std.testing.expect(!defaultRewind(true, false, true, 128, 0));
    try std.testing.expect(!defaultRewind(true, true, false, 128, 0));
    try std.testing.expect(!defaultRewind(true, true, true, 0, 0));
    try std.testing.expect(!defaultRewind(true, true, true, 128, 4096));
    try std.testing.expect(rewindRequested(null, true));
    try std.testing.expect(!rewindRequested(null, false));
    for ([_]bool{ true, false }) |default| {
        try std.testing.expect(rewindRequested("1", default));
        for ([_][]const u8{ "0", "", "true", "false", "unknown", " 1" }) |setting| {
            try std.testing.expect(!rewindRequested(setting, default));
        }
    }
}

test "prefix rewind replaces a dropped thinking header and later tool history" {
    // The template drops the generated empty thought channel and reconstructs
    // the tool call. Reusing rows after the first difference would retain the
    // wrong positions; replaying the entire system prompt wastes valid KV.
    const stored = [_]u32{ 1, 2, 3, 90, 91, 4, 5, 6 };
    const canonical = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try std.testing.expectEqual(@as(usize, 3), rewindPrefixLen(&stored, &canonical));
    const edited_tool = [_]u32{ 1, 2, 3, 90, 91, 4, 99, 8 };
    try std.testing.expectEqual(@as(usize, 6), rewindPrefixLen(&stored, &edited_tool));
    try std.testing.expectEqual(@as(usize, 0), rewindPrefixLen(&stored, &.{ 99, 2, 3 }));
}

test "prefix rewind always leaves a token to refresh logits" {
    const stored = [_]u32{ 1, 2, 3 };
    try std.testing.expectEqual(@as(usize, 2), rewindPrefixLen(&stored, &stored));
    try std.testing.expectEqual(@as(usize, 1), rewindPrefixLen(&stored, &.{ 1, 2 }));
    try std.testing.expectEqual(@as(usize, 3), rewindPrefixLen(&stored, &.{ 1, 2, 3, 4 }));
    try std.testing.expectEqual(@as(usize, 0), rewindPrefixLen(&stored, &.{1}));
    try std.testing.expectEqual(@as(usize, 0), rewindPrefixLen(&stored, &.{}));
    try std.testing.expectEqual(@as(usize, 0), rewindPrefixLen(&.{}, &stored));
}
