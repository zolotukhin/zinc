//! Generated struct-layout probe used by the site Zig API docs.
//!
//! The docs build emits a temporary Zig file that imports selected public
//! structs and prints their size, alignment, and field offsets; this runner
//! simply dispatches to that generated probe.
const generated = @import(".zig-api-cache/zig-struct-analyzer.generated.zig");

pub fn main() !void {
    try generated.main();
}
