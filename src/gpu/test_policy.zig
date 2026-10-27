//! Explicit hardware-test opt-in for dedicated GPU workers.
//!
//! The build enables these cases only for `-Dgpu-tests=true` or full mode.
//! Default unit tests skip before device creation, allocation, or dispatch.
const std = @import("std");
const host_io = @import("host_io");

fn enabledValue(value: ?[]const u8) bool {
    return if (value) |text| std.mem.eql(u8, text, "1") else false;
}

/// Whether this test invocation explicitly allows GPU initialization.
pub fn enabled() bool {
    return enabledValue(host_io.getenv("ZINC_RUN_GPU_TESTS"));
}

/// Skip a hardware test before it creates any GPU context or buffer.
pub fn requireGpuTests() error{SkipZigTest}!void {
    if (!enabled()) return error.SkipZigTest;
}

test "hardware test opt-in accepts only the literal one" {
    try std.testing.expect(enabledValue("1"));
    for ([_]?[]const u8{ null, "", "0", "true", "01", "1 ", "1extra" }) |value| {
        try std.testing.expect(!enabledValue(value));
    }
}
