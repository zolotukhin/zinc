//! Select fixed-input-width projection variants from existing GGUF metadata.
//! @section Shader Dispatch
//! This module is host-only so its selection contracts run on every backend.
const std = @import("std");
const TensorInfo = @import("../model/gguf.zig").TensorInfo;

const specialized_projection_widths = [_]u32{ 2816, 4096, 5120, 5376, 6144, 12288, 17408, 21504 };

/// Fixed-K variants are useful only when a model has a matrix with that input
/// width. Creating every variant retains unrelated driver code and compilation
/// state for the model's lifetime. Generic pipelines remain available, and
/// unknown future specialization widths load conservatively until listed here.
pub const ProjectionSpecializations = struct {
    present: std.StaticBitSet(specialized_projection_widths.len) = .empty,

    /// Retain every variant for standalone kernel benchmarks.
    pub fn all() ProjectionSpecializations {
        return .{ .present = .full };
    }

    /// Record the input width of a matrix or expert tensor.
    pub fn includeTensor(self: *ProjectionSpecializations, tensor: TensorInfo) void {
        // GGUF matrices are innermost-first: dims[0] is the dot-product K.
        // Output rows, expert counts and norm vectors do not establish a K.
        if (tensor.n_dims < 2) return;
        for (specialized_projection_widths, 0..) |width, index| {
            if (tensor.dims[0] == width) self.present.set(index);
        }
    }

    /// Whether a fixed-K pipeline can be used by the loaded model.
    pub fn needs(self: ProjectionSpecializations, width: u32) bool {
        const index = std.mem.indexOfScalar(u32, &specialized_projection_widths, width) orelse return true;
        return self.present.isSet(index);
    }
};

test "projection demand follows matrix input widths rather than rows or experts" {
    var demand: ProjectionSpecializations = .{};
    var tensor: TensorInfo = .{ .name = "fixture", .n_dims = 2, .dims = .{ 4096, 21504, 1, 1 }, .type_ = .q4_k, .offset = 0 };
    demand.includeTensor(tensor);
    try std.testing.expect(demand.needs(4096));
    try std.testing.expect(!demand.needs(21504));
    tensor.n_dims = 3;
    tensor.dims = .{ 12288, 4096, 6144, 1 };
    demand.includeTensor(tensor);
    try std.testing.expect(demand.needs(12288));
    try std.testing.expect(!demand.needs(6144));
    tensor.n_dims = 1;
    tensor.dims = .{ 5376, 1, 1, 1 };
    demand.includeTensor(tensor);
    try std.testing.expect(!demand.needs(5376));
    tensor.n_dims = 2;
    tensor.dims = .{ 21504, 5376, 1, 1 };
    demand.includeTensor(tensor);
    try std.testing.expect(demand.needs(21504));
    try std.testing.expect(!demand.needs(17408));
    try std.testing.expect(demand.needs(7777)); // Unlisted future variants stay available.
    for (specialized_projection_widths) |width| try std.testing.expect(ProjectionSpecializations.all().needs(width));
}
