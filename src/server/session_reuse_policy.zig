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

test "automatic reuse excludes recurrent state and unqualified backends" {
    try std.testing.expect(canAdvertise(true, 0));
    try std.testing.expect(!canAdvertise(true, 4096));
    try std.testing.expect(!canAdvertise(false, 0));
    try std.testing.expect(!canAdvertise(false, 4096));
}
