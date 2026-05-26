//! Shape-static ZINC_RT IR graph builder.
//! Graphs contain logical buffers and opcode nodes before any tier lowers them
//! into packets, shaders, or pure Zig calls.
//! @section Decode Planning
const std = @import("std");
const op = @import("op.zig");

/// Dense index identifying a logical buffer inside a Graph.
/// Buffer ids are assigned sequentially by `Graph.addBuffer` starting at 0.
pub const BufferId = u32;

/// Dense index identifying a node (opcode invocation) inside a Graph.
/// Node ids are assigned in insertion order by `Graph.addNode`.
pub const NodeId = u32;

/// Maximum number of input or output buffers a single node may bind.
/// Picked to keep `BindingList` inline-storable without heap allocation.
pub const max_bindings = 8;

/// Fixed-capacity list of buffer ids bound to one side of a node.
/// Stores up to `max_bindings` entries inline so a `Node` stays POD and
/// trivially copyable through the planner.
pub const BindingList = struct {
    items: [max_bindings]BufferId = undefined,
    len: u8 = 0,

    /// Build a binding list from a slice of buffer ids.
    /// @param values Buffer ids to copy; must contain at most `max_bindings`.
    /// @returns A fully populated `BindingList`.
    /// @note Returns `error.TooManyBindings` when `values.len > max_bindings`.
    pub fn init(values: []const BufferId) !BindingList {
        if (values.len > max_bindings) return error.TooManyBindings;
        var result = BindingList{};
        result.len = @intCast(values.len);
        for (values, 0..) |value, index| {
            result.items[index] = value;
        }
        return result;
    }

    /// View the populated prefix of the binding list as a slice.
    /// @returns The first `self.len` buffer ids, in insertion order.
    pub fn slice(self: *const BindingList) []const BufferId {
        return self.items[0..self.len];
    }
};

/// Single opcode invocation in the graph.
/// Each node references its inputs and outputs by `BufferId`; the opcode
/// itself decides the semantics of those bindings (see `op.Info`).
pub const Node = struct {
    opcode: op.Opcode,
    inputs: BindingList,
    outputs: BindingList,
};

/// Shape-static ZINC_RT IR graph.
/// A graph is a flat list of opcode nodes plus a buffer count; lowering
/// passes turn this representation into T-CPU packets, PM4 indirect buffers,
/// or Metal/Vulkan dispatches without mutating the graph itself.
pub const Graph = struct {
    allocator: std.mem.Allocator,
    buffers: u32 = 0,
    nodes: std.ArrayList(Node) = .empty,

    /// Create an empty graph backed by `allocator`.
    /// @param allocator Used for the node array; not retained for buffer storage.
    /// @returns A zero-buffer, zero-node graph ready for `addBuffer`/`addNode`.
    pub fn init(allocator: std.mem.Allocator) Graph {
        return .{ .allocator = allocator };
    }

    /// Release the node array and poison the graph value.
    /// @note Buffer ids are integers and require no per-buffer release.
    pub fn deinit(self: *Graph) void {
        self.nodes.deinit(self.allocator);
        self.* = undefined;
    }

    /// Reserve a new logical buffer and return its id.
    /// @returns The freshly allocated `BufferId`, equal to the previous buffer count.
    pub fn addBuffer(self: *Graph) BufferId {
        const id = self.buffers;
        self.buffers += 1;
        return id;
    }

    /// Append an opcode node with the given input and output bindings.
    /// @param opcode Opcode this node executes.
    /// @param inputs Buffer ids consumed by the node, in opcode-defined order.
    /// @param outputs Buffer ids produced by the node, in opcode-defined order.
    /// @returns The `NodeId` of the newly appended node.
    /// @note Fails with `error.TooManyBindings` when either slice exceeds `max_bindings`.
    pub fn addNode(
        self: *Graph,
        opcode: op.Opcode,
        inputs: []const BufferId,
        outputs: []const BufferId,
    ) !NodeId {
        const id: NodeId = @intCast(self.nodes.items.len);
        try self.nodes.append(self.allocator, .{
            .opcode = opcode,
            .inputs = try BindingList.init(inputs),
            .outputs = try BindingList.init(outputs),
        });
        return id;
    }

    /// Structural sanity check: graph is non-empty, all bindings resolve,
    /// and every non-barrier/stream_out node produces at least one output.
    /// @returns `error.EmptyGraph`, `error.UnknownInputBuffer`,
    /// `error.UnknownOutputBuffer`, or `error.NodeWithoutOutput` on failure.
    pub fn verify(self: *const Graph) !void {
        if (self.nodes.items.len == 0) return error.EmptyGraph;

        for (self.nodes.items) |node| {
            for (node.inputs.slice()) |buffer| {
                if (buffer >= self.buffers) return error.UnknownInputBuffer;
            }
            for (node.outputs.slice()) |buffer| {
                if (buffer >= self.buffers) return error.UnknownOutputBuffer;
            }
            if (node.outputs.len == 0 and node.opcode != .barrier and node.opcode != .stream_out) {
                return error.NodeWithoutOutput;
            }
        }
    }
};

test "graph rejects unknown buffers" {
    var graph = Graph.init(std.testing.allocator);
    defer graph.deinit();

    const input = graph.addBuffer();
    const output = graph.addBuffer();
    _ = try graph.addNode(.rms_norm, &.{ input, 99 }, &.{output});
    try std.testing.expectError(error.UnknownInputBuffer, graph.verify());
}
