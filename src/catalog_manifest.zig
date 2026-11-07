//! Export managed-model catalog metadata without GPU access.
//!
//! Coverage checks compile this entrypoint and compare its JSON with the Intel
//! release policy. It serializes catalog constants without initializing a GPU
//! or opening a GGUF file.
//! @section CLI & Entrypoints
const std = @import("std");
const catalog = @import("model/catalog.zig");

/// Write the compiled catalog as JSON to standard output.
pub fn main(process: std.process.Init) !void {
    const json = try std.json.Stringify.valueAlloc(process.gpa, catalog.entries, .{});
    defer process.gpa.free(json);
    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(process.io, &buffer);
    try stdout.interface.writeAll(json);
    try stdout.interface.writeByte('\n');
    try stdout.interface.flush();
}
