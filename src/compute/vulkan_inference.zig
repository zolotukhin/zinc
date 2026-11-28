//! Vulkan engine entry points shared by the CLI and HTTP server.
//! @section Inference Runtime
//!
//! Dispatches to the standard transformer engine or the Inkling loop decoder while keeping sampling and request state shared.
const std = @import("std");
const host_io = @import("host_io");
const standard = @import("forward.zig");
const inkling = @import("forward_inkling_vulkan.zig");
const memory_plan = @import("../gpu/memory_plan.zig");
const loader = @import("../model/loader.zig");
const Instance = @import("../vulkan/instance.zig").Instance;
const GpuConfig = @import("../vulkan/gpu_detect.zig").GpuConfig;
const Graph = @import("graph.zig").Graph;

const log = std.log.scoped(.forward);
pub const DecodeState = standard.DecodeState;
pub const SamplingParams = standard.SamplingParams;
pub const tensorBytes = standard.tensorBytes;

pub const InferenceEngine = union(enum) {
    standard: standard.InferenceEngine,
    inkling: inkling.Forward,

    pub const mtp_max_draft_tokens = standard.InferenceEngine.mtp_max_draft_tokens;

    pub fn init(model: *loader.Model, instance: *const Instance, config: GpuConfig, shader_dir: []const u8, allocator: std.mem.Allocator) !InferenceEngine {
        if (model.config.architecture == .inkling2_loop)
            return .{ .inkling = try inkling.Forward.init(model, instance, shader_dir, allocator) };
        return .{ .standard = try standard.InferenceEngine.init(model, instance, config, shader_dir, allocator) };
    }

    pub fn deinit(self: *InferenceEngine) void {
        switch (self.*) {
            inline else => |*engine| engine.deinit(),
        }
    }

    pub fn contextCapacity(self: *const InferenceEngine) u32 {
        return switch (self.*) {
            .standard => |*engine| engine.max_context_tokens,
            .inkling => |*engine| engine.max_ctx,
        };
    }

    pub fn decodeGraph(self: *const InferenceEngine) ?*const Graph {
        return switch (self.*) {
            .standard => |*engine| &engine.decode_graph,
            .inkling => null,
        };
    }

    pub fn enableProfiling(self: *InferenceEngine) !void {
        switch (self.*) {
            .standard => |*engine| try engine.enableProfiling(),
            .inkling => return error.ProfilingUnsupportedForInkling,
        }
    }

    pub fn enableValidationDiagnostics(self: *InferenceEngine) void {
        if (self.* == .standard) self.standard.enableValidationDiagnostics();
    }

    pub fn enableLogitsReadback(self: *InferenceEngine) void {
        self.setLogitsReadbackEnabled(true);
    }

    pub fn setLogitsReadbackEnabled(self: *InferenceEngine, enabled: bool) void {
        if (self.* == .standard) self.standard.logits_readback_enabled = enabled;
    }

    pub fn logitsReadbackEnabled(self: *const InferenceEngine) bool {
        return switch (self.*) {
            .standard => |*engine| engine.logits_readback_enabled,
            .inkling => true,
        };
    }

    pub fn lastLogits(self: *const InferenceEngine) []const f32 {
        return switch (self.*) {
            .standard => |*engine| blk: {
                const ptr: [*]const f32 = @ptrCast(@alignCast(engine.logits_staging.mapped.?));
                break :blk ptr[0..engine.model.config.vocab_size];
            },
            .inkling => |*engine| engine.logits(),
        };
    }

    pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {
        switch (self.*) {
            .standard => |*engine| try engine.prefillBatched(state, prompt_tokens),
            .inkling => |*engine| {
                try engine.prefill(prompt_tokens, state.position);
                state.position += @intCast(prompt_tokens.len);
            },
        }
    }

    pub fn decodeStep(self: *InferenceEngine, state: *DecodeState, token_id: u32, collect_output: bool) !void {
        switch (self.*) {
            .standard => |*engine| try engine.decodeStep(state, token_id, collect_output),
            .inkling => |*engine| {
                try engine.step(token_id, state.position, collect_output);
                state.position += 1;
            },
        }
    }

    pub fn sample(self: *const InferenceEngine, state: *const DecodeState, params: SamplingParams, random: std.Random) u32 {
        return switch (self.*) {
            .standard => |*engine| engine.sample(state, params, random),
            .inkling => |*engine| if (params.requiresLogitsReadback())
                standard.InferenceEngine.sampleFromLogits(engine.logits(), state.generated_tokens.items, params, random, engine.d.logit_cap, 1.0)
            else
                engine.greedy(),
        };
    }
};

pub fn generate(engine: *InferenceEngine, prompt_tokens: []const u32, max_tokens: u32, eos: u32, extra_stop_ids: []const u32, allocator: std.mem.Allocator) ![]u32 {
    switch (engine.*) {
        .standard => |*decoder| return standard.generate(decoder, prompt_tokens, max_tokens, eos, extra_stop_ids, allocator),
        .inkling => |*decoder| {
            if (prompt_tokens.len == 0) return error.EmptyPrompt;
            if (prompt_tokens.len > decoder.max_ctx) return error.ContextLengthExceeded;
            const budget = memory_plan.requestBudget(@intCast(prompt_tokens.len), max_tokens, decoder.max_ctx);
            const prefill_start = host_io.nanoTimestamp();
            try decoder.prefill(prompt_tokens, 0);
            const prefill_end = host_io.nanoTimestamp();
            const prefill_ms = milliseconds(prefill_start, prefill_end);
            log.info("Prefill: {d} tokens in {d:.1} ms ({d:.2} tok/s)", .{
                prompt_tokens.len, prefill_ms, @as(f64, @floatFromInt(prompt_tokens.len)) * 1000.0 / prefill_ms,
            });

            var tokens: std.ArrayList(u32) = .empty;
            errdefer tokens.deinit(allocator);
            var current = decoder.greedy();
            var position: u32 = @intCast(prompt_tokens.len);
            for (0..budget.completion_tokens) |i| {
                if (current == eos or std.mem.indexOfScalar(u32, extra_stop_ids, current) != null) break;
                try tokens.append(allocator, current);
                if (i + 1 == budget.completion_tokens) break;
                try decoder.step(current, position, true);
                current = decoder.greedy();
                position += 1;
            }
            const decode_ms = milliseconds(prefill_end, host_io.nanoTimestamp());
            const generated: f64 = @floatFromInt(tokens.items.len);
            log.info("Generated {d} tokens in {d:.1} ms — {d:.2} tok/s ({d:.1} ms/tok)", .{
                tokens.items.len, decode_ms, generated * 1000.0 / decode_ms, decode_ms / @max(1.0, generated),
            });
            return tokens.toOwnedSlice(allocator);
        },
    }
}

fn milliseconds(start: i128, end: i128) f64 {
    return @as(f64, @floatFromInt(@max(1, end - start))) / 1_000_000.0;
}
