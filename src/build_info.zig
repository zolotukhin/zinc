//! Build metadata exported by `build.zig` for CLI version reporting.
//! @section CLI & Entrypoints
//!
//! The values in this module are generated at compile time from build options
//! such as `-Dversion` and `-Dcommit`, then printed by `zinc --version`.
const std = @import("std");
const build_options = @import("build_options");

/// Semantic version string for this build (from `-Dversion`, e.g. `0.3.1`).
pub const version = build_options.version;
/// Short git commit hash this binary was built from (from `-Dcommit`).
pub const commit = build_options.commit;
/// Compilation target triple this binary was built for (from `-Dtarget`).
pub const target = build_options.target;
/// Active optimize mode, e.g. `ReleaseFast` or `Debug` (from `-Doptimize`).
pub const optimize = build_options.optimize;
/// Single GPU backend compiled into this binary, e.g. `vulkan`, `rocm`, or `metal`.
pub const backend = build_options.backend;

/// Write the full `zinc --version` report to `writer`.
///
/// Emits the version, commit, target, optimize mode, and compiled-in backend,
/// each on its own line. The `backends` output key is retained for compatibility.
/// @param writer Destination writer that receives the formatted metadata block.
/// @returns Propagates only the writer's own error if printing fails.
pub fn writeVersion(writer: anytype) !void {
    try writer.print(
        \\zinc {s}
        \\commit: {s}
        \\target: {s}
        \\optimize: {s}
        \\backends: {s}
        \\
    , .{
        version,
        commit,
        target,
        optimize,
        backend,
    });
}

test "version metadata is present" {
    try std.testing.expect(version.len > 0);
    try std.testing.expect(commit.len > 0);
    try std.testing.expect(target.len > 0);
    try std.testing.expect(optimize.len > 0);
    try std.testing.expect(backend.len > 0);
}
