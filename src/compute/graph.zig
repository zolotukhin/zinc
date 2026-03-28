//! Represent decode work as a dependency graph that can be topologically ordered.
//! @section Decode Planning
//! Graph builders use this module to describe fused operations, dependencies,
//! and dispatch metadata before any Vulkan command recording happens.
const std = @import("std");

const log = std.log.scoped(.graph);

/// Operation types that a compute graph node can represent.
///
/// Each variant maps to one GPU shader dispatch or fused kernel invocation
/// during decode-time execution.
pub const OpType = enum {
    /// Dense matrix multiply used during prefill with cooperative matrices.
    matmul,
    /// Decode-time matrix-vector product for single-token projection.
    dmmv,

    /// Fused RMS normalization followed by element-wise scale multiply.
    rms_norm_mul,
    /// SwiGLU activation: SiLU(x) * y.
    swiglu,
    /// Sigmoid gating: sigmoid(x) * y, used by SSM layers.
    sigmoid_mul,
    /// Rotary position embedding with reshape and KV-cache write.
    rope,
    /// Row-wise softmax over attention scores.
    softmax,
    /// Softmax followed by top-k selection for MoE expert routing.
    softmax_topk,

    /// Paged flash attention with grouped-query attention (GQA) support.
    flash_attn,

    /// Write key/value vectors into the paged KV cache.
    kv_cache_write,
    /// Read key/value vectors from the paged KV cache.
    kv_cache_read,

    /// TurboQuant: quantize key vectors for compressed KV storage.
    tq_compress_keys,
    /// TurboQuant: quantize value vectors for compressed KV storage.
    tq_compress_values,
    /// TurboQuant: asymmetric attention computed over compressed KV pairs.
    tq_attention,
    /// TurboQuant: decompress values and perform weighted accumulation.
    tq_decompress_values,

    /// MoE expert routing gate that selects active experts per token.
    moe_gate,
    /// Gather and combine outputs from the selected MoE experts.
    moe_gather,

    /// Element-wise vector addition.
    add,
    /// Raw buffer-to-buffer copy.
    copy,
    /// Token embedding table lookup.
    embed,
};

/// A single operation node in the compute dependency graph.
///
/// Each node carries dispatch metadata (workgroups, push constants) and
/// dependency edges so the graph can be topologically sorted before recording.
pub const Node = struct {
    /// Unique dense identifier assigned in insertion order.
    id: u32,
    /// Operation type this node performs when dispatched.
    op: OpType,
    /// Human-readable label used in logs and diagnostic output.
    name: []const u8,

    /// Buffer table indices consumed by the shader (up to 4 input buffers).
    inputs: [4]?u32,
    /// Buffer table index produced by the shader, if any.
    output: ?u32,
    /// Number of valid entries in the `inputs` array.
    n_inputs: u8,

    /// Workgroup counts in the x, y, z dimensions for compute dispatch.
    workgroup_count: [3]u32,
    /// Raw push constant payload copied into the command buffer at dispatch time.
    push_constants: [64]u8,
    /// Number of bytes actually used in `push_constants`.
    push_constant_size: u8,

    /// Index into the pipeline table, or null when not yet assigned.
    pipeline_index: ?u32,
    /// Node IDs that must complete before this node may execute (up to 8).
    depends_on: [8]?u32,
    /// Number of valid entries in the `depends_on` array.
    n_deps: u8,
};

/// Directed dependency edge between two graph nodes.
pub const Edge = struct {
    /// Producer node ID.
    from_id: u32,
    /// Consumer node ID.
    to_id: u32,
};

/// Count of nodes that share the same operation type.
pub const OpCount = struct {
    /// Operation type.
    op: OpType,
    /// Number of occurrences.
    count: u32,
};

/// Critical-path node annotated with its dependency depth.
pub const CriticalPathNode = struct {
    /// Unique identifier.
    id: u32,
    /// Name identifier.
    name: []const u8,
    /// Operation type.
    op: OpType,
    /// Depth from nearest root.
    depth: u32,
};

/// Per-node structural metrics derived from the dependency graph.
pub const NodeAnalysis = struct {
    /// Unique identifier.
    id: u32,
    /// Name identifier.
    name: []const u8,
    /// Operation type.
    op: OpType,
    /// Upstream dependencies.
    dependency_count: u32,
    dependent_count: u32,
    /// Depth from nearest root.
    depth: u32,
    /// True if no dependencies.
    is_root: bool,
    /// True if nothing depends on this.
    is_leaf: bool,
    is_on_critical_path: bool,
    workgroups: [3]u32,
};

/// Computed summary of the graph structure used by visualization and debugging tools.
pub const GraphAnalysis = struct {
    /// Name identifier.
    name: []const u8,
    /// Total nodes in graph.
    node_count: u32,
    edge_count: u32,
    root_count: u32,
    leaf_count: u32,
    max_depth: u32,
    critical_path_node_count: u32,
    critical_path_edge_count: u32,
    max_parallel_width: u32,
    depth_widths: []u32,
    op_counts: []OpCount,
    critical_path: []CriticalPathNode,
    /// Graph nodes.
    nodes: []NodeAnalysis,
    edges: []Edge,
    /// Allocator for owned resources.
    allocator: std.mem.Allocator,

    /// Release the arrays allocated for the analysis result.
    /// @param self Graph analysis to tear down in place.
    pub fn deinit(self: *GraphAnalysis) void {
        self.allocator.free(self.depth_widths);
        self.allocator.free(self.op_counts);
        self.allocator.free(self.critical_path);
        self.allocator.free(self.nodes);
        self.allocator.free(self.edges);
        self.* = undefined;
    }
};

/// Static compute graph for a transformer layer or full decode pass.
pub const Graph = struct {
    /// Graph nodes.
    nodes: std.ArrayList(Node) = .{},
    /// Allocator for owned resources.
    allocator: std.mem.Allocator,
    /// Name identifier.
    name: []const u8,

    /// Initialize an empty graph with a human-readable name.
    /// @param allocator Allocator used for node storage.
    /// @param name Debug name for logging and diagnostics.
    /// @returns A graph ready to accept nodes and dependencies.
    pub fn init(allocator: std.mem.Allocator, name: []const u8) Graph {
        return Graph{
            .allocator = allocator,
            .name = name,
        };
    }

    /// Release all graph nodes owned by the graph.
    /// @param self Graph to tear down in place.
    pub fn deinit(self: *Graph) void {
        self.nodes.deinit(self.allocator);
        self.* = undefined;
    }

    /// Append a node to the graph and assign it the next dense node ID.
    /// @param self Graph to append to.
    /// @param op Operation kind represented by the new node.
    /// @param name Human-readable node label used in logs and diagnostics.
    /// @returns The node ID assigned to the appended node.
    /// @note IDs are stable for the lifetime of the graph and match insertion order.
    pub fn addNode(self: *Graph, op: OpType, name: []const u8) !u32 {
        const id: u32 = @intCast(self.nodes.items.len);
        try self.nodes.append(self.allocator, .{
            .id = id,
            .op = op,
            .name = name,
            .inputs = .{ null, null, null, null },
            .output = null,
            .n_inputs = 0,
            .workgroup_count = .{ 1, 1, 1 },
            .push_constants = undefined,
            .push_constant_size = 0,
            .pipeline_index = null,
            .depends_on = .{ null, null, null, null, null, null, null, null },
            .n_deps = 0,
        });
        return id;
    }

    /// Set the input buffer table indices consumed by a node.
    /// @param self Graph containing the node to update.
    /// @param node_id ID of the node whose inputs should be overwritten.
    /// @param inputs Buffer table indices consumed in shader binding order.
    /// @note The slice is copied into the node's fixed-size input array.
    pub fn setInputs(self: *Graph, node_id: u32, inputs: []const u32) void {
        var node = &self.nodes.items[node_id];
        for (inputs, 0..) |buf, i| {
            node.inputs[i] = buf;
        }
        node.n_inputs = @intCast(inputs.len);
    }

    /// Set the output buffer table index produced by a node.
    /// @param self Graph containing the node to update.
    /// @param node_id ID of the node whose output should be overwritten.
    /// @param output Buffer table index produced by the node.
    pub fn setOutput(self: *Graph, node_id: u32, output: u32) void {
        self.nodes.items[node_id].output = output;
    }

    /// Set the workgroup dimensions that should be used when dispatching a node.
    /// @param self Graph containing the node to update.
    /// @param node_id ID of the node whose dispatch dimensions should be overwritten.
    /// @param x Workgroup count in the X dimension.
    /// @param y Workgroup count in the Y dimension.
    /// @param z Workgroup count in the Z dimension.
    pub fn setWorkgroups(self: *Graph, node_id: u32, x: u32, y: u32, z: u32) void {
        self.nodes.items[node_id].workgroup_count = .{ x, y, z };
    }

    /// Declare that one node must execute after another.
    /// @param self Graph containing both nodes.
    /// @param node_id Node that depends on `depends_on`.
    /// @param depends_on Node that must run first.
    /// @note Cycles are not rejected here; `topologicalOrder()` detects them later.
    pub fn addDependency(self: *Graph, node_id: u32, depends_on: u32) void {
        var node = &self.nodes.items[node_id];
        node.depends_on[node.n_deps] = depends_on;
        node.n_deps += 1;
    }

    /// Compute a valid execution order for the current dependency graph.
    /// @param self Graph to sort.
    /// @param allocator Allocator used for temporary in-degree tracking and the returned order slice.
    /// @returns Node IDs in a valid execution order, or `error.CyclicDependency` when the graph contains a cycle.
    pub fn topologicalOrder(self: *const Graph, allocator: std.mem.Allocator) ![]u32 {
        const n = self.nodes.items.len;
        if (n == 0) return try allocator.alloc(u32, 0);

        var in_degree = try allocator.alloc(u32, n);
        defer allocator.free(in_degree);
        @memset(in_degree, 0);

        // In-degree = number of dependencies for each node
        for (self.nodes.items, 0..) |node, i| {
            in_degree[i] = node.n_deps;
        }

        // Kahn's algorithm
        var queue: std.ArrayList(u32) = .{};
        defer queue.deinit(allocator);

        for (0..n) |i| {
            if (in_degree[i] == 0) try queue.append(allocator, @intCast(i));
        }

        var result = try allocator.alloc(u32, n);
        var result_idx: usize = 0;

        while (queue.items.len > 0) {
            const current = queue.orderedRemove(0);
            result[result_idx] = current;
            result_idx += 1;

            // For each node that depends on current, decrease in-degree
            for (self.nodes.items, 0..) |node, i| {
                for (node.depends_on[0..node.n_deps]) |dep_opt| {
                    if (dep_opt) |dep| {
                        if (dep == current) {
                            in_degree[i] -= 1;
                            if (in_degree[i] == 0) {
                                try queue.append(allocator, @intCast(i));
                            }
                        }
                    }
                }
            }
        }

        if (result_idx != n) {
            allocator.free(result);
            return error.CyclicDependency;
        }

        return result;
    }

    /// Return the number of nodes currently stored in the graph.
    /// @param self Graph to inspect.
    /// @returns The number of appended nodes.
    pub fn nodeCount(self: *const Graph) usize {
        return self.nodes.items.len;
    }

    /// Analyze dependency structure for visualization and optimization work.
    /// @param self Graph to inspect.
    /// @param allocator Allocator used for the returned analysis arrays.
    /// @returns A GraphAnalysis containing op counts, edges, node depths, and the longest dependency chain.
    pub fn analyze(self: *const Graph, allocator: std.mem.Allocator) !GraphAnalysis {
        const n = self.nodes.items.len;
        const node_count: u32 = @intCast(n);
        if (n == 0) {
            return GraphAnalysis{
                .name = self.name,
                .node_count = 0,
                .edge_count = 0,
                .root_count = 0,
                .leaf_count = 0,
                .max_depth = 0,
                .critical_path_node_count = 0,
                .critical_path_edge_count = 0,
                .max_parallel_width = 0,
                .depth_widths = try allocator.alloc(u32, 0),
                .op_counts = try allocator.alloc(OpCount, 0),
                .critical_path = try allocator.alloc(CriticalPathNode, 0),
                .nodes = try allocator.alloc(NodeAnalysis, 0),
                .edges = try allocator.alloc(Edge, 0),
                .allocator = allocator,
            };
        }

        const topo = try self.topologicalOrder(allocator);
        defer allocator.free(topo);

        var dependency_depths = try allocator.alloc(u32, n);
        defer allocator.free(dependency_depths);
        @memset(dependency_depths, 0);

        var parent_on_critical_path = try allocator.alloc(u32, n);
        defer allocator.free(parent_on_critical_path);
        @memset(parent_on_critical_path, std.math.maxInt(u32));

        var dependent_counts = try allocator.alloc(u32, n);
        defer allocator.free(dependent_counts);
        @memset(dependent_counts, 0);

        var edge_count: u32 = 0;
        var root_count: u32 = 0;
        for (self.nodes.items) |node| {
            edge_count += node.n_deps;
            if (node.n_deps == 0) root_count += 1;
            for (node.depends_on[0..node.n_deps]) |dep_opt| {
                if (dep_opt) |dep| dependent_counts[dep] += 1;
            }
        }

        var max_depth: u32 = 0;
        var critical_end: u32 = topo[0];
        for (topo) |node_id| {
            const node = self.nodes.items[node_id];
            var best_parent = parent_on_critical_path[node_id];
            var best_depth = dependency_depths[node_id];

            for (node.depends_on[0..node.n_deps]) |dep_opt| {
                if (dep_opt) |dep| {
                    const candidate_depth = dependency_depths[dep] + 1;
                    if (candidate_depth > best_depth) {
                        best_depth = candidate_depth;
                        best_parent = dep;
                    }
                }
            }

            dependency_depths[node_id] = best_depth;
            parent_on_critical_path[node_id] = best_parent;
            if (best_depth > max_depth) {
                max_depth = best_depth;
                critical_end = node_id;
            }
        }

        var depth_widths = try allocator.alloc(u32, max_depth + 1);
        errdefer allocator.free(depth_widths);
        @memset(depth_widths, 0);

        var leaf_count: u32 = 0;
        for (dependency_depths, 0..) |depth, idx| {
            depth_widths[depth] += 1;
            if (dependent_counts[idx] == 0) leaf_count += 1;
        }

        var max_parallel_width: u32 = 0;
        for (depth_widths) |width| {
            if (width > max_parallel_width) max_parallel_width = width;
        }

        const op_fields = std.meta.fields(OpType);
        var raw_op_counts = try allocator.alloc(u32, op_fields.len);
        defer allocator.free(raw_op_counts);
        @memset(raw_op_counts, 0);
        for (self.nodes.items) |node| {
            raw_op_counts[@intFromEnum(node.op)] += 1;
        }

        var nonzero_op_count: usize = 0;
        for (raw_op_counts) |count| {
            if (count > 0) nonzero_op_count += 1;
        }

        var op_counts = try allocator.alloc(OpCount, nonzero_op_count);
        errdefer allocator.free(op_counts);
        var op_idx: usize = 0;
        for (raw_op_counts, 0..) |count, index| {
            if (count == 0) continue;
            op_counts[op_idx] = .{
                .op = @enumFromInt(index),
                .count = count,
            };
            op_idx += 1;
        }
        for (0..op_counts.len) |i| {
            for (i + 1..op_counts.len) |j| {
                if (op_counts[j].count > op_counts[i].count) {
                    const tmp = op_counts[i];
                    op_counts[i] = op_counts[j];
                    op_counts[j] = tmp;
                }
            }
        }

        const critical_path_node_count = max_depth + 1;
        var critical_path = try allocator.alloc(CriticalPathNode, critical_path_node_count);
        errdefer allocator.free(critical_path);

        var critical_mask = try allocator.alloc(bool, n);
        defer allocator.free(critical_mask);
        @memset(critical_mask, false);

        var cursor = critical_end;
        var reverse_index: usize = critical_path.len;
        while (true) {
            reverse_index -= 1;
            const node = self.nodes.items[cursor];
            critical_mask[cursor] = true;
            critical_path[reverse_index] = .{
                .id = cursor,
                .name = node.name,
                .op = node.op,
                .depth = dependency_depths[cursor],
            };

            const parent = parent_on_critical_path[cursor];
            if (parent == std.math.maxInt(u32)) break;
            cursor = parent;
        }

        var nodes = try allocator.alloc(NodeAnalysis, n);
        errdefer allocator.free(nodes);
        for (self.nodes.items, 0..) |node, idx| {
            nodes[idx] = .{
                .id = node.id,
                .name = node.name,
                .op = node.op,
                .dependency_count = node.n_deps,
                .dependent_count = dependent_counts[idx],
                .depth = dependency_depths[idx],
                .is_root = node.n_deps == 0,
                .is_leaf = dependent_counts[idx] == 0,
                .is_on_critical_path = critical_mask[idx],
                .workgroups = node.workgroup_count,
            };
        }

        var edges = try allocator.alloc(Edge, edge_count);
        errdefer allocator.free(edges);
        var edge_idx: usize = 0;
        for (self.nodes.items) |node| {
            for (node.depends_on[0..node.n_deps]) |dep_opt| {
                if (dep_opt) |dep| {
                    edges[edge_idx] = .{
                        .from_id = dep,
                        .to_id = node.id,
                    };
                    edge_idx += 1;
                }
            }
        }

        return GraphAnalysis{
            .name = self.name,
            .node_count = node_count,
            .edge_count = edge_count,
            .root_count = root_count,
            .leaf_count = leaf_count,
            .max_depth = max_depth,
            .critical_path_node_count = critical_path_node_count,
            .critical_path_edge_count = max_depth,
            .max_parallel_width = max_parallel_width,
            .depth_widths = depth_widths,
            .op_counts = op_counts,
            .critical_path = critical_path,
            .nodes = nodes,
            .edges = edges,
            .allocator = allocator,
        };
    }

    /// Serialize a graph-analysis JSON payload suitable for custom viewers and scripts.
    /// @param self Graph to inspect and serialize.
    /// @param writer Destination writer for the JSON payload.
    /// @param allocator Allocator used for temporary analysis storage.
    pub fn writeJsonReport(self: *const Graph, writer: *std.Io.Writer, allocator: std.mem.Allocator) !void {
        var analysis = try self.analyze(allocator);
        defer analysis.deinit();

        try std.json.Stringify.value(.{
            .name = analysis.name,
            .node_count = analysis.node_count,
            .edge_count = analysis.edge_count,
            .root_count = analysis.root_count,
            .leaf_count = analysis.leaf_count,
            .max_depth = analysis.max_depth,
            .critical_path_node_count = analysis.critical_path_node_count,
            .critical_path_edge_count = analysis.critical_path_edge_count,
            .max_parallel_width = analysis.max_parallel_width,
            .depth_widths = analysis.depth_widths,
            .op_counts = analysis.op_counts,
            .critical_path = analysis.critical_path,
            .nodes = analysis.nodes,
            .edges = analysis.edges,
        }, .{ .whitespace = .indent_2 }, writer);
        try writer.writeByte('\n');
    }

    /// Serialize the graph as Graphviz DOT for quick local rendering.
    /// @param self Graph to inspect and serialize.
    /// @param writer Destination writer for the DOT payload.
    /// @param allocator Allocator used for temporary analysis storage.
    pub fn writeDot(self: *const Graph, writer: *std.Io.Writer, allocator: std.mem.Allocator) !void {
        var analysis = try self.analyze(allocator);
        defer analysis.deinit();

        try writer.writeAll("digraph zinc_decode {\n");
        try writer.writeAll("  rankdir=LR;\n");
        try writer.writeAll("  graph [fontname=\"Menlo\", labelloc=\"t\"];\n");
        try writer.writeAll("  node [shape=box, style=\"rounded\", fontname=\"Menlo\"];\n");
        try writer.writeAll("  edge [fontname=\"Menlo\"];\n");
        try writer.print("  label=\"{s}: {d} nodes, {d} edges, critical path {d} nodes\";\n", .{
            analysis.name,
            analysis.node_count,
            analysis.edge_count,
            analysis.critical_path_node_count,
        });

        for (analysis.nodes) |node| {
            const critical_attrs = if (node.is_on_critical_path)
                ", color=\"#b91c1c\", penwidth=2, fillcolor=\"#fee2e2\", style=\"rounded,filled\""
            else
                "";
            try writer.print("  n{d} [label=\"{d}: {s}\\n{s}\\ndepth={d}\"{s}];\n", .{
                node.id,
                node.id,
                @tagName(node.op),
                node.name,
                node.depth,
                critical_attrs,
            });
        }

        for (analysis.edges) |edge| {
            try writer.print("  n{d} -> n{d};\n", .{
                edge.from_id,
                edge.to_id,
            });
        }

        try writer.writeAll("}\n");
    }
};

test "Graph: basic add and topo sort" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator, "test");
    defer g.deinit();

    const n0 = try g.addNode(.embed, "embed");
    const n1 = try g.addNode(.rms_norm_mul, "norm1");
    const n2 = try g.addNode(.dmmv, "qkv_proj");

    g.addDependency(n1, n0);
    g.addDependency(n2, n1);

    const order = try g.topologicalOrder(allocator);
    defer allocator.free(order);

    try std.testing.expectEqual(@as(usize, 3), order.len);
    try std.testing.expectEqual(n0, order[0]);
    try std.testing.expectEqual(n1, order[1]);
    try std.testing.expectEqual(n2, order[2]);
}

test "Graph: parallel nodes" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator, "test");
    defer g.deinit();

    const n0 = try g.addNode(.embed, "embed");
    _ = try g.addNode(.dmmv, "q_proj"); // n1, no deps
    _ = try g.addNode(.dmmv, "k_proj"); // n2, no deps
    const n3 = try g.addNode(.flash_attn, "attn");

    g.addDependency(n3, n0);

    const order = try g.topologicalOrder(allocator);
    defer allocator.free(order);

    try std.testing.expectEqual(@as(usize, 4), order.len);
    // n3 must come after n0
    var n0_pos: usize = 0;
    var n3_pos: usize = 0;
    for (order, 0..) |id, i| {
        if (id == n0) n0_pos = i;
        if (id == n3) n3_pos = i;
    }
    try std.testing.expect(n0_pos < n3_pos);
}

test "Graph: analyze returns op counts and critical path" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator, "analysis");
    defer g.deinit();

    const embed = try g.addNode(.embed, "embed");
    const norm = try g.addNode(.rms_norm_mul, "norm");
    const q_proj = try g.addNode(.dmmv, "q_proj");
    const k_proj = try g.addNode(.dmmv, "k_proj");
    const attn = try g.addNode(.flash_attn, "attn");

    g.addDependency(norm, embed);
    g.addDependency(q_proj, norm);
    g.addDependency(k_proj, norm);
    g.addDependency(attn, q_proj);
    g.addDependency(attn, k_proj);

    var analysis = try g.analyze(allocator);
    defer analysis.deinit();

    try std.testing.expectEqual(@as(u32, 5), analysis.node_count);
    try std.testing.expectEqual(@as(u32, 5), analysis.edge_count);
    try std.testing.expectEqual(@as(u32, 1), analysis.root_count);
    try std.testing.expectEqual(@as(u32, 1), analysis.leaf_count);
    try std.testing.expectEqual(@as(u32, 3), analysis.max_depth);
    try std.testing.expectEqual(@as(u32, 4), analysis.critical_path_node_count);
    try std.testing.expectEqual(@as(u32, 2), analysis.max_parallel_width);
    try std.testing.expectEqualStrings("embed", analysis.critical_path[0].name);
    try std.testing.expectEqualStrings("attn", analysis.critical_path[analysis.critical_path.len - 1].name);

    var dmmv_count: ?u32 = null;
    for (analysis.op_counts) |entry| {
        if (entry.op == .dmmv) dmmv_count = entry.count;
    }
    try std.testing.expectEqual(@as(?u32, 2), dmmv_count);
}

test "Graph: dot and json exports include node labels" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator, "export");
    defer g.deinit();

    const a = try g.addNode(.embed, "embed");
    const b = try g.addNode(.dmmv, "proj");
    g.addDependency(b, a);

    var json_buf: std.Io.Writer.Allocating = .init(allocator);
    defer json_buf.deinit();
    try g.writeJsonReport(&json_buf.writer, allocator);
    try std.testing.expect(std.mem.indexOf(u8, json_buf.written(), "\"critical_path\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_buf.written(), "\"op\": \"dmmv\"") != null);

    var dot_buf: std.Io.Writer.Allocating = .init(allocator);
    defer dot_buf.deinit();
    try g.writeDot(&dot_buf.writer, allocator);
    try std.testing.expect(std.mem.indexOf(u8, dot_buf.written(), "digraph zinc_decode") != null);
    try std.testing.expect(std.mem.indexOf(u8, dot_buf.written(), "n0 -> n1") != null);
}
