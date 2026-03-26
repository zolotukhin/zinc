const std = @import("std");

const log = std.log.scoped(.graph);

/// Operation types in the compute graph.
pub const OpType = enum {
    // Matrix operations
    matmul, // Dense matrix multiply (prefill, cooperative matrix)
    dmmv, // Decode matmul-vec (single token)

    // Element-wise fused operations
    rms_norm_mul, // RMS normalization + scale multiply
    swiglu, // SiLU(x) * y
    sigmoid_mul, // sigmoid(x) * y (SSM gating)
    rope, // Rotary position embedding + reshape + cache write
    softmax, // Softmax (for attention scores)
    softmax_topk, // Softmax + top-k (MoE routing)

    // Attention
    flash_attn, // Paged flash attention with GQA

    // KV cache
    kv_cache_write, // Write K/V to paged cache
    kv_cache_read, // Read K/V from paged cache

    // TurboQuant
    tq_compress_keys, // Quantize keys
    tq_compress_values, // Quantize values
    tq_attention, // Asymmetric attention on compressed KV
    tq_decompress_values, // Decompress + weighted accumulation

    // MoE
    moe_gate, // Expert routing
    moe_gather, // Gather expert outputs

    // Utility
    add, // Element-wise add
    copy, // Buffer copy
    embed, // Token embedding lookup
};

/// A node in the compute graph.
pub const Node = struct {
    id: u32,
    op: OpType,
    name: []const u8,

    // Input/output buffer references (indices into a buffer table)
    inputs: [4]?u32, // up to 4 input buffers
    output: ?u32, // output buffer
    n_inputs: u8,

    // Dispatch parameters
    workgroup_count: [3]u32, // x, y, z
    push_constants: [64]u8, // raw push constant data
    push_constant_size: u8,

    // Execution metadata
    pipeline_index: ?u32, // index into pipeline table
    depends_on: [8]?u32, // node IDs this depends on
    n_deps: u8,
};

/// Static compute graph for a transformer layer or full decode pass.
pub const Graph = struct {
    nodes: std.ArrayList(Node) = .{},
    allocator: std.mem.Allocator,
    name: []const u8,

    pub fn init(allocator: std.mem.Allocator, name: []const u8) Graph {
        return Graph{
            .allocator = allocator,
            .name = name,
        };
    }

    pub fn deinit(self: *Graph) void {
        self.nodes.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add a node to the graph and return its ID.
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

    /// Set input buffers for a node.
    pub fn setInputs(self: *Graph, node_id: u32, inputs: []const u32) void {
        var node = &self.nodes.items[node_id];
        for (inputs, 0..) |buf, i| {
            node.inputs[i] = buf;
        }
        node.n_inputs = @intCast(inputs.len);
    }

    /// Set output buffer for a node.
    pub fn setOutput(self: *Graph, node_id: u32, output: u32) void {
        self.nodes.items[node_id].output = output;
    }

    /// Set workgroup dispatch dimensions.
    pub fn setWorkgroups(self: *Graph, node_id: u32, x: u32, y: u32, z: u32) void {
        self.nodes.items[node_id].workgroup_count = .{ x, y, z };
    }

    /// Add a dependency between nodes.
    pub fn addDependency(self: *Graph, node_id: u32, depends_on: u32) void {
        var node = &self.nodes.items[node_id];
        node.depends_on[node.n_deps] = depends_on;
        node.n_deps += 1;
    }

    /// Get a topologically sorted execution order.
    /// Returns node indices in valid execution order.
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

    /// Number of nodes.
    pub fn nodeCount(self: *const Graph) usize {
        return self.nodes.items.len;
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
