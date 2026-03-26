//! Run the inference runtime: decode state, pipeline ownership, and token generation.
//! @section Inference Runtime
//! This module ties together model state, compute graphs, dispatch helpers,
//! and greedy token sampling for a single active inference engine.
const std = @import("std");
const vk = @import("../vulkan/vk.zig");
const Instance = @import("../vulkan/instance.zig").Instance;
const Buffer = @import("../vulkan/buffer.zig").Buffer;
const CommandPool = @import("../vulkan/command.zig").CommandPool;
const CommandBuffer = @import("../vulkan/command.zig").CommandBuffer;
const GpuConfig = @import("../vulkan/gpu_detect.zig").GpuConfig;
const loader = @import("../model/loader.zig");
const Model = loader.Model;
const ModelConfig = loader.ModelConfig;
const architecture = @import("../model/architecture.zig");
const Graph = @import("graph.zig").Graph;
const DmmvDispatch = @import("dmmv.zig").DmmvDispatch;
const ElementwiseDispatch = @import("elementwise.zig").ElementwiseDispatch;
const AttentionDispatch = @import("attention.zig").AttentionDispatch;

const log = std.log.scoped(.forward);

/// Runtime state for the decode loop.
pub const DecodeState = struct {
    position: u32,
    generated_tokens: std.ArrayList(u32),
    allocator: std.mem.Allocator,

    /// Initialize decode state for a fresh generation request.
    /// @param allocator Allocator used for the generated token buffer.
    /// @returns A DecodeState positioned at token index zero.
    pub fn init(allocator: std.mem.Allocator) DecodeState {
        return .{
            .position = 0,
            .generated_tokens = .{},
            .allocator = allocator,
        };
    }

    /// Release the generated token buffer owned by the decode state.
    /// @param self Decode state to tear down in place.
    pub fn deinit(self: *DecodeState) void {
        self.generated_tokens.deinit(self.allocator);
        self.* = undefined;
    }
};

/// Inference engine combining model, pipelines, and dispatch.
pub const InferenceEngine = struct {
    model: *Model,
    gpu_config: GpuConfig,
    dmmv: DmmvDispatch,
    elementwise: ElementwiseDispatch,
    attention: AttentionDispatch,
    cmd_pool: CommandPool,
    decode_cmd: CommandBuffer,
    decode_graph: Graph,
    // Intermediate buffers for decode
    hidden_buf: Buffer, // hidden state buffer
    residual_buf: Buffer, // residual connection buffer
    logits_buf: Buffer, // output logits
    instance: *const Instance,
    allocator: std.mem.Allocator,

    /// Create the runtime objects needed to execute decode-time work on the GPU.
    /// @param model Loaded model weights and metadata.
    /// @param instance Active Vulkan instance and logical device.
    /// @param gpu_config Derived GPU tuning parameters.
    /// @param shader_dir Directory containing compiled SPIR-V shader binaries.
    /// @param allocator Allocator used for graphs and temporary runtime state.
    /// @returns An initialized inference engine ready to record decode work.
    pub fn init(
        model: *Model,
        instance: *const Instance,
        gpu_config: GpuConfig,
        shader_dir: []const u8,
        allocator: std.mem.Allocator,
    ) !InferenceEngine {
        const config = &model.config;

        var cmd_pool = try CommandPool.init(instance);
        errdefer cmd_pool.deinit();

        var decode_cmd = try CommandBuffer.init(instance, &cmd_pool);
        errdefer decode_cmd.deinit(&cmd_pool);

        var dmmv = try DmmvDispatch.init(instance, &gpu_config, shader_dir, allocator);
        errdefer dmmv.deinit();

        var elementwise = try ElementwiseDispatch.init(instance, shader_dir, allocator);
        errdefer elementwise.deinit();

        var attention = try AttentionDispatch.init(instance, shader_dir, allocator);
        errdefer attention.deinit();

        // Build the decode graph
        var decode_graph = try architecture.buildDecodeGraph(config, allocator);
        errdefer decode_graph.deinit();

        // Allocate intermediate buffers
        const hidden_size = @as(vk.c.VkDeviceSize, config.hidden_dim) * @sizeOf(f32);
        var hidden_buf = try Buffer.initDeviceLocal(instance, hidden_size, vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        errdefer hidden_buf.deinit();

        var residual_buf = try Buffer.initDeviceLocal(instance, hidden_size, vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        errdefer residual_buf.deinit();

        const logits_size = @as(vk.c.VkDeviceSize, config.vocab_size) * @sizeOf(f32);
        var logits_buf = try Buffer.initDeviceLocal(instance, logits_size, vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        errdefer logits_buf.deinit();

        log.info("Inference engine ready — {d} graph nodes, hidden_dim={d}, vocab={d}", .{
            decode_graph.nodeCount(), config.hidden_dim, config.vocab_size,
        });

        return InferenceEngine{
            .model = model,
            .gpu_config = gpu_config,
            .dmmv = dmmv,
            .elementwise = elementwise,
            .attention = attention,
            .cmd_pool = cmd_pool,
            .decode_cmd = decode_cmd,
            .decode_graph = decode_graph,
            .hidden_buf = hidden_buf,
            .residual_buf = residual_buf,
            .logits_buf = logits_buf,
            .instance = instance,
            .allocator = allocator,
        };
    }

    /// Run a single decode step: process one token position and produce logits.
    /// Returns the index into the logits buffer (for sampling).
    pub fn decodeStep(self: *InferenceEngine, state: *DecodeState) !void {
        // Record command buffer for this decode step
        try self.decode_cmd.reset();
        try self.decode_cmd.begin();

        // Walk the graph in topological order and record dispatches
        const order = try self.decode_graph.topologicalOrder(self.allocator);
        defer self.allocator.free(order);

        for (order) |node_id| {
            const node = &self.decode_graph.nodes.items[node_id];

            switch (node.op) {
                .rms_norm_mul => {
                    self.decode_cmd.computeBarrier();
                    // Dispatch will be fully wired when descriptor sets are managed
                },
                .dmmv => {
                    self.decode_cmd.computeBarrier();
                },
                .rope => {},
                .kv_cache_write => {},
                .flash_attn => {
                    self.decode_cmd.computeBarrier();
                },
                .swiglu => {},
                .add => {},
                .embed => {},
                else => {},
            }
        }

        try self.decode_cmd.end();
        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

        state.position += 1;
    }

    /// Sample a token from the logits buffer (greedy argmax for now).
    pub fn sampleGreedy(self: *InferenceEngine) !u32 {
        // Read logits back to CPU
        const vocab_size = self.model.config.vocab_size;
        const logits_size = @as(usize, vocab_size) * @sizeOf(f32);

        var staging = try Buffer.initStaging(self.instance, logits_size);
        defer staging.deinit();

        const buffer_mod = @import("../vulkan/buffer.zig");
        try buffer_mod.copyBuffer(self.instance, self.cmd_pool.handle, &self.logits_buf, &staging, logits_size);

        const logits_ptr: [*]const f32 = @ptrCast(@alignCast(staging.mapped.?));
        const logits = logits_ptr[0..vocab_size];

        // Argmax
        var max_val: f32 = logits[0];
        var max_idx: u32 = 0;
        for (logits[1..], 1..) |val, i| {
            if (val > max_val) {
                max_val = val;
                max_idx = @intCast(i);
            }
        }

        return max_idx;
    }

    /// Release GPU buffers, graphs, and dispatch helpers owned by the engine.
    /// @param self Inference engine to tear down in place.
    pub fn deinit(self: *InferenceEngine) void {
        self.logits_buf.deinit();
        self.residual_buf.deinit();
        self.hidden_buf.deinit();
        self.decode_graph.deinit();
        self.attention.deinit();
        self.elementwise.deinit();
        self.dmmv.deinit();
        self.decode_cmd.deinit(&self.cmd_pool);
        self.cmd_pool.deinit();
        self.* = undefined;
    }
};

/// Run single-request inference: tokenize → prefill → decode → detokenize.
pub fn generate(
    engine: *InferenceEngine,
    prompt_tokens: []const u32,
    max_tokens: u32,
    allocator: std.mem.Allocator,
) ![]u32 {
    var state = DecodeState.init(allocator);
    defer state.deinit();

    log.info("Generating: {d} prompt tokens, max {d} output tokens", .{
        prompt_tokens.len, max_tokens,
    });

    // Prefill: process all prompt tokens
    for (prompt_tokens) |_| {
        try engine.decodeStep(&state);
    }

    log.info("Prefill complete at position {d}", .{state.position});

    // Decode: generate tokens one at a time
    var generated: u32 = 0;
    while (generated < max_tokens) : (generated += 1) {
        try engine.decodeStep(&state);
        const token = try engine.sampleGreedy();
        try state.generated_tokens.append(allocator, token);

        // Check for EOS (token 0 or 2 are common EOS tokens)
        if (token == 0 or token == 2) break;
    }

    log.info("Generated {d} tokens", .{state.generated_tokens.items.len});

    return try allocator.dupe(u32, state.generated_tokens.items);
}
