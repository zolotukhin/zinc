//! Backend-specific server runtime aliases and wrappers.
//! Keeps the HTTP/routes layer shared across Vulkan and Metal backends.
const std = @import("std");
const gpu = @import("../gpu/interface.zig");

pub const is_metal = gpu.is_metal;
pub const is_vulkan = gpu.is_vulkan;
pub const supports_model_management = gpu.is_vulkan or gpu.is_metal;
pub const supports_sampling_controls = gpu.is_vulkan;
pub const supports_runtime_profiling = gpu.is_vulkan;

pub const tokenizer_mod = @import("../model/tokenizer.zig");
pub const forward_mod = if (gpu.is_metal) @import("../compute/forward_metal.zig") else @import("../compute/forward.zig");
pub const loader_mod = if (gpu.is_metal) @import("../model/loader_metal.zig") else @import("../model/loader.zig");
pub const model_manager_mod = if (gpu.is_metal) @import("model_manager_metal.zig") else @import("model_manager.zig");

pub const InferenceEngine = forward_mod.InferenceEngine;
pub const DecodeState = forward_mod.DecodeState;
pub const Model = loader_mod.Model;
pub const ModelManager = model_manager_mod.ModelManager;

pub const SamplingParams = if (gpu.is_metal) struct {
    temperature: f32 = 0.0,
    top_p: f32 = 1.0,
    repetition_penalty: f32 = 1.0,
    top_k: u32 = 0,

    pub fn requiresLogitsReadback(self: @This()) bool {
        _ = self;
        return false;
    }
} else forward_mod.SamplingParams;

pub fn enableLogitsReadback(_engine: *InferenceEngine) void {
    if (comptime gpu.is_vulkan) {
        _engine.enableLogitsReadback();
    }
}

pub fn logitsReadbackEnabled(_engine: *const InferenceEngine) bool {
    if (comptime gpu.is_vulkan) {
        return _engine.logits_readback_enabled;
    }
    return false;
}

pub fn setLogitsReadbackEnabled(_engine: *InferenceEngine, _enabled: bool) void {
    if (comptime gpu.is_vulkan) {
        _engine.logits_readback_enabled = _enabled;
    }
}

pub fn enableProfiling(_engine: *InferenceEngine) !void {
    if (comptime gpu.is_vulkan) {
        try _engine.enableProfiling();
    }
}

pub fn decodeStep(
    _engine: *InferenceEngine,
    _state: *DecodeState,
    _token_id: u32,
    _collect_output: bool,
) !void {
    if (comptime gpu.is_vulkan) {
        try _engine.decodeStep(_state, _token_id, _collect_output);
    } else {
        try _engine.decodeStep(_state, _token_id);
    }
}

pub fn sample(
    _engine: *const InferenceEngine,
    _state: *const DecodeState,
    _params: SamplingParams,
    _random: std.Random,
) u32 {
    if (comptime gpu.is_vulkan) {
        return _engine.sample(_state, _params, _random);
    }
    return _engine.sampleGreedy();
}
