//! Represent decode work as a dependency graph that can be topologically ordered.
//! @section Decode Planning
//! Graph builders use this module to describe fused operations, dependencies,
//! and dispatch metadata before any Vulkan command recording happens.
const std = @import("std");

const log = std.log.scoped(.graph);

pub const ExecDomain = enum {
    gpu_compute,
    gpu_transfer,
    cpu_host,
};

pub const BottleneckKind = enum {
    memory_bandwidth,
    occupancy,
    launch_latency,
    host_sync,
    transfer,
    compute,
    unknown,
};

pub const HardwareInfo = struct {
    bandwidth_gbps: u32 = 0,
    compute_units: u32 = 0,
    wave_size: u32 = 64,
    preferred_workgroup_size: u32 = 64,
};

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
    /// Decode layer index when the node is layer-local.
    layer_index: ?u32,
    /// Where the work executes.
    domain: ExecDomain,

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
    /// Threads launched per workgroup for occupancy estimates.
    threads_per_workgroup: u32,
    /// Estimated activation reads per dispatch.
    read_bytes: u64,
    /// Estimated writes per dispatch.
    write_bytes: u64,
    /// Estimated weight / tensor payload bytes streamed by the op.
    weight_bytes: u64,
    /// Approximate floating-point work.
    flops: u64,
    /// True when this node forces host-visible synchronization or readback.
    requires_host_sync: bool,
    /// Optional static note for diagnostics.
    note: ?[]const u8,

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
    layer_index: ?u32,
    domain: ExecDomain,
    workgroups: [3]u32,
    threads_per_workgroup: u32,
    total_workgroups: u64,
    read_bytes: u64,
    write_bytes: u64,
    weight_bytes: u64,
    total_bytes: u64,
    flops: u64,
    arithmetic_intensity: f64,
    estimated_bandwidth_time_us: ?f64,
    bandwidth_ceiling_pct: ?f64,
    bottleneck: BottleneckKind,
    requires_host_sync: bool,
    bottleneck_reason: []const u8,
    note: ?[]const u8,
};

pub const Hotspot = struct {
    id: u32,
    name: []const u8,
    op: OpType,
    layer_index: ?u32,
    estimated_share_pct: f64,
    estimated_bandwidth_time_us: ?f64,
    total_bytes: u64,
    flops: u64,
    bottleneck: BottleneckKind,
    bottleneck_reason: []const u8,
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
    assumed_decode_seq_len: u32,
    hardware: HardwareInfo,
    total_read_bytes: u64,
    total_write_bytes: u64,
    total_weight_bytes: u64,
    total_bytes: u64,
    total_flops: u64,
    depth_widths: []u32,
    op_counts: []OpCount,
    critical_path: []CriticalPathNode,
    hotspots: []Hotspot,
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
        self.allocator.free(self.hotspots);
        self.allocator.free(self.nodes);
        self.allocator.free(self.edges);
        self.* = undefined;
    }
};

fn totalWorkgroups(workgroups: [3]u32) u64 {
    return @as(u64, workgroups[0]) * @as(u64, workgroups[1]) * @as(u64, workgroups[2]);
}

fn totalBytes(node: *const Node) u64 {
    return node.read_bytes + node.write_bytes + node.weight_bytes;
}

fn arithmeticIntensity(node: *const Node) f64 {
    const bytes = totalBytes(node);
    if (bytes == 0) return 0.0;
    return @as(f64, @floatFromInt(node.flops)) / @as(f64, @floatFromInt(bytes));
}

fn estimateBandwidthTimeUs(bytes: u64, bandwidth_gbps: u32) ?f64 {
    if (bytes == 0 or bandwidth_gbps == 0) return null;
    const bytes_per_sec = @as(f64, @floatFromInt(bandwidth_gbps)) * 1_000_000_000.0;
    return @as(f64, @floatFromInt(bytes)) / bytes_per_sec * 1_000_000.0;
}

fn estimateBandwidthCeilingPct(node: *const Node, hardware: HardwareInfo) ?f64 {
    if (node.domain != .gpu_compute or hardware.compute_units == 0 or hardware.wave_size == 0) return null;

    const total_wgs = totalWorkgroups(node.workgroup_count);
    if (total_wgs == 0) return 0.0;

    const waves_per_workgroup = @max(@as(u64, 1), std.math.divCeil(u64, node.threads_per_workgroup, hardware.wave_size) catch 1);
    const total_waves = total_wgs * waves_per_workgroup;
    const target_waves = @as(f64, @floatFromInt(hardware.compute_units)) * 4.0;
    if (target_waves <= 0.0) return null;
    return @min(100.0, @as(f64, @floatFromInt(total_waves)) / target_waves * 100.0);
}

fn classifyNode(node: *const Node, hardware: HardwareInfo) struct { kind: BottleneckKind, reason: []const u8 } {
    const wg_total = totalWorkgroups(node.workgroup_count);
    const bytes = totalBytes(node);
    const intensity = arithmeticIntensity(node);
    const ceiling = estimateBandwidthCeilingPct(node, hardware);

    if (node.requires_host_sync or node.domain == .cpu_host) {
        return .{
            .kind = .host_sync,
            .reason = "CPU-visible work or synchronization breaks queue overlap.",
        };
    }
    if (node.domain == .gpu_transfer) {
        return .{
            .kind = .transfer,
            .reason = "Copy/readback traffic moves bytes but does not saturate compute units.",
        };
    }
    if (ceiling != null and ceiling.? < 35.0) {
        return .{
            .kind = .occupancy,
            .reason = "Too few waves/workgroups to keep most CUs busy.",
        };
    }
    if (wg_total <= 8 or (bytes <= 256 * 1024 and wg_total <= 32)) {
        return .{
            .kind = .launch_latency,
            .reason = "Dispatch is too small to ramp memory bandwidth to peak.",
        };
    }
    if (intensity < 2.0) {
        return .{
            .kind = .memory_bandwidth,
            .reason = "Low arithmetic intensity means memory traffic dominates runtime.",
        };
    }
    if (intensity >= 8.0) {
        return .{
            .kind = .compute,
            .reason = "Higher arithmetic intensity may shift the bottleneck toward ALU/reduction work.",
        };
    }
    return .{
        .kind = .unknown,
        .reason = "Mixed characteristics; inspect bytes, workgroups, and dependencies together.",
    };
}

/// Static compute graph for a transformer layer or full decode pass.
pub const Graph = struct {
    /// Graph nodes.
    nodes: std.ArrayList(Node) = .{},
    /// Allocator for owned resources.
    allocator: std.mem.Allocator,
    /// Name identifier.
    name: []const u8,
    /// Sequence length assumed by decode-time cost estimates.
    assumed_decode_seq_len: u32 = 0,
    /// Optional hardware context used for utilization heuristics.
    hardware: HardwareInfo = .{},

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
        for (self.nodes.items) |node| self.allocator.free(node.name);
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
        const owned_name = try self.allocator.dupe(u8, name);
        try self.nodes.append(self.allocator, .{
            .id = id,
            .op = op,
            .name = owned_name,
            .layer_index = null,
            .domain = .gpu_compute,
            .inputs = .{ null, null, null, null },
            .output = null,
            .n_inputs = 0,
            .workgroup_count = .{ 1, 1, 1 },
            .push_constants = undefined,
            .push_constant_size = 0,
            .threads_per_workgroup = 64,
            .read_bytes = 0,
            .write_bytes = 0,
            .weight_bytes = 0,
            .flops = 0,
            .requires_host_sync = false,
            .note = null,
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

    pub fn setLayerIndex(self: *Graph, node_id: u32, layer_index: ?u32) void {
        self.nodes.items[node_id].layer_index = layer_index;
    }

    pub fn setExecDomain(self: *Graph, node_id: u32, domain: ExecDomain) void {
        self.nodes.items[node_id].domain = domain;
    }

    pub fn setThreadsPerWorkgroup(self: *Graph, node_id: u32, threads_per_workgroup: u32) void {
        self.nodes.items[node_id].threads_per_workgroup = threads_per_workgroup;
    }

    pub fn setCostEstimate(self: *Graph, node_id: u32, read_bytes: u64, write_bytes: u64, weight_bytes: u64, flops: u64) void {
        var node = &self.nodes.items[node_id];
        node.read_bytes = read_bytes;
        node.write_bytes = write_bytes;
        node.weight_bytes = weight_bytes;
        node.flops = flops;
    }

    pub fn setHostSync(self: *Graph, node_id: u32, requires_host_sync: bool) void {
        self.nodes.items[node_id].requires_host_sync = requires_host_sync;
    }

    pub fn setNote(self: *Graph, node_id: u32, note: ?[]const u8) void {
        self.nodes.items[node_id].note = note;
    }

    pub fn setAssumedDecodeSeqLen(self: *Graph, assumed_decode_seq_len: u32) void {
        self.assumed_decode_seq_len = assumed_decode_seq_len;
    }

    pub fn setHardwareContext(self: *Graph, hardware: HardwareInfo) void {
        self.hardware = hardware;
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
                .assumed_decode_seq_len = self.assumed_decode_seq_len,
                .hardware = self.hardware,
                .total_read_bytes = 0,
                .total_write_bytes = 0,
                .total_weight_bytes = 0,
                .total_bytes = 0,
                .total_flops = 0,
                .depth_widths = try allocator.alloc(u32, 0),
                .op_counts = try allocator.alloc(OpCount, 0),
                .critical_path = try allocator.alloc(CriticalPathNode, 0),
                .hotspots = try allocator.alloc(Hotspot, 0),
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
        var total_read_bytes_: u64 = 0;
        var total_write_bytes_: u64 = 0;
        var total_weight_bytes_: u64 = 0;
        var total_flops_: u64 = 0;
        for (self.nodes.items, 0..) |node, idx| {
            const node_total_bytes = totalBytes(&node);
            const node_total_workgroups = totalWorkgroups(node.workgroup_count);
            const node_intensity = arithmeticIntensity(&node);
            const bw_time_us = estimateBandwidthTimeUs(node_total_bytes, self.hardware.bandwidth_gbps);
            const bw_ceiling_pct = estimateBandwidthCeilingPct(&node, self.hardware);
            const bottleneck = classifyNode(&node, self.hardware);

            total_read_bytes_ += node.read_bytes;
            total_write_bytes_ += node.write_bytes;
            total_weight_bytes_ += node.weight_bytes;
            total_flops_ += node.flops;

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
                .layer_index = node.layer_index,
                .domain = node.domain,
                .workgroups = node.workgroup_count,
                .threads_per_workgroup = node.threads_per_workgroup,
                .total_workgroups = node_total_workgroups,
                .read_bytes = node.read_bytes,
                .write_bytes = node.write_bytes,
                .weight_bytes = node.weight_bytes,
                .total_bytes = node_total_bytes,
                .flops = node.flops,
                .arithmetic_intensity = node_intensity,
                .estimated_bandwidth_time_us = bw_time_us,
                .bandwidth_ceiling_pct = bw_ceiling_pct,
                .bottleneck = bottleneck.kind,
                .requires_host_sync = node.requires_host_sync,
                .bottleneck_reason = bottleneck.reason,
                .note = node.note,
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

        const total_bytes_ = total_read_bytes_ + total_write_bytes_ + total_weight_bytes_;
        const hotspot_count = @min(n, 12);
        var hotspot_indices = try allocator.alloc(usize, n);
        defer allocator.free(hotspot_indices);
        for (0..n) |idx| hotspot_indices[idx] = idx;

        const has_hw_score = self.hardware.bandwidth_gbps > 0;
        const total_score: f64 = blk: {
            var score: f64 = 0.0;
            for (nodes) |node| {
                score += if (has_hw_score)
                    (node.estimated_bandwidth_time_us orelse 0.0)
                else
                    @as(f64, @floatFromInt(node.total_bytes));
            }
            break :blk score;
        };

        for (0..hotspot_indices.len) |i| {
            for (i + 1..hotspot_indices.len) |j| {
                const lhs = nodes[hotspot_indices[i]];
                const rhs = nodes[hotspot_indices[j]];
                const lhs_score = if (has_hw_score)
                    (lhs.estimated_bandwidth_time_us orelse 0.0)
                else
                    @as(f64, @floatFromInt(lhs.total_bytes));
                const rhs_score = if (has_hw_score)
                    (rhs.estimated_bandwidth_time_us orelse 0.0)
                else
                    @as(f64, @floatFromInt(rhs.total_bytes));
                if (rhs_score > lhs_score) {
                    const tmp = hotspot_indices[i];
                    hotspot_indices[i] = hotspot_indices[j];
                    hotspot_indices[j] = tmp;
                }
            }
        }

        var hotspots = try allocator.alloc(Hotspot, hotspot_count);
        errdefer allocator.free(hotspots);
        for (0..hotspot_count) |i| {
            const node = nodes[hotspot_indices[i]];
            const score = if (has_hw_score)
                (node.estimated_bandwidth_time_us orelse 0.0)
            else
                @as(f64, @floatFromInt(node.total_bytes));
            hotspots[i] = .{
                .id = node.id,
                .name = node.name,
                .op = node.op,
                .layer_index = node.layer_index,
                .estimated_share_pct = if (total_score > 0.0) score / total_score * 100.0 else 0.0,
                .estimated_bandwidth_time_us = node.estimated_bandwidth_time_us,
                .total_bytes = node.total_bytes,
                .flops = node.flops,
                .bottleneck = node.bottleneck,
                .bottleneck_reason = node.bottleneck_reason,
            };
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
            .assumed_decode_seq_len = self.assumed_decode_seq_len,
            .hardware = self.hardware,
            .total_read_bytes = total_read_bytes_,
            .total_write_bytes = total_write_bytes_,
            .total_weight_bytes = total_weight_bytes_,
            .total_bytes = total_bytes_,
            .total_flops = total_flops_,
            .depth_widths = depth_widths,
            .op_counts = op_counts,
            .critical_path = critical_path,
            .hotspots = hotspots,
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
            .assumed_decode_seq_len = analysis.assumed_decode_seq_len,
            .hardware = analysis.hardware,
            .total_read_bytes = analysis.total_read_bytes,
            .total_write_bytes = analysis.total_write_bytes,
            .total_weight_bytes = analysis.total_weight_bytes,
            .total_bytes = analysis.total_bytes,
            .total_flops = analysis.total_flops,
            .depth_widths = analysis.depth_widths,
            .op_counts = analysis.op_counts,
            .critical_path = analysis.critical_path,
            .hotspots = analysis.hotspots,
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
        try writer.print("  label=\"{s}: {d} nodes, {d} edges, critical path {d} nodes, ~{d:.1} MB/token\";\n", .{
            analysis.name,
            analysis.node_count,
            analysis.edge_count,
            analysis.critical_path_node_count,
            @as(f64, @floatFromInt(analysis.total_bytes)) / 1_000_000.0,
        });

        for (analysis.nodes) |node| {
            const critical_attrs = if (node.is_on_critical_path)
                ", color=\"#b91c1c\", penwidth=2, fillcolor=\"#fee2e2\", style=\"rounded,filled\""
            else
                "";
            try writer.print("  n{d} [label=\"{d}: {s}\\n{s}\\ndepth={d} wg={d}\\n{d:.2} MB {s}\"{s}];\n", .{
                node.id,
                node.id,
                @tagName(node.op),
                node.name,
                node.depth,
                node.total_workgroups,
                @as(f64, @floatFromInt(node.total_bytes)) / 1_000_000.0,
                @tagName(node.bottleneck),
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

test "Graph: performance analysis ranks hotspots and occupancy" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator, "perf");
    defer g.deinit();

    g.setHardwareContext(.{
        .bandwidth_gbps = 500,
        .compute_units = 64,
        .wave_size = 64,
        .preferred_workgroup_size = 64,
    });

    const tiny = try g.addNode(.dmmv, "tiny_proj");
    g.setWorkgroups(tiny, 2, 1, 1);
    g.setCostEstimate(tiny, 4 * 1024, 4 * 1024, 64 * 1024, 64 * 1024);

    const heavy = try g.addNode(.dmmv, "lm_head");
    g.setWorkgroups(heavy, 256, 1, 1);
    g.setCostEstimate(heavy, 16 * 1024, 512 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024);
    g.addDependency(heavy, tiny);

    var analysis = try g.analyze(allocator);
    defer analysis.deinit();

    try std.testing.expect(analysis.hotspots.len >= 1);
    try std.testing.expectEqualStrings("lm_head", analysis.hotspots[0].name);
    try std.testing.expectEqual(BottleneckKind.occupancy, analysis.nodes[0].bottleneck);
    try std.testing.expect(analysis.nodes[1].estimated_bandwidth_time_us != null);
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
    try std.testing.expect(std.mem.indexOf(u8, json_buf.written(), "\"hotspots\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_buf.written(), "\"op\": \"dmmv\"") != null);

    var dot_buf: std.Io.Writer.Allocating = .init(allocator);
    defer dot_buf.deinit();
    try g.writeDot(&dot_buf.writer, allocator);
    try std.testing.expect(std.mem.indexOf(u8, dot_buf.written(), "digraph zinc_decode") != null);
    try std.testing.expect(std.mem.indexOf(u8, dot_buf.written(), "n0 -> n1") != null);
}
