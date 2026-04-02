const generated = @import(".zig-api-cache/zig-struct-analyzer.generated.zig");

pub fn main() !void {
    try generated.main();
}
