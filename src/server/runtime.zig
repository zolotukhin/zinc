//! Backend-specific server runtime aliases and wrappers.
//! Keeps the HTTP/routes layer shared across Vulkan and Metal backends.
//! @section API Server
const std = @import("std");
const gpu = @import("../gpu/interface.zig");

/// Whether the active GPU backend is Apple Metal.
pub const is_metal = gpu.is_metal;
/// Whether the active GPU backend is Vulkan.
pub const is_vulkan = gpu.is_vulkan;
/// Whether the backend supports loading/unloading models at runtime.
pub const supports_model_management = gpu.is_vulkan or gpu.is_metal;
/// Whether the backend supports temperature, top-p, top-k, and repetition penalty.
pub const supports_sampling_controls = gpu.is_vulkan or gpu.is_metal;
/// Whether the backend supports GPU kernel profiling during inference.
pub const supports_runtime_profiling = gpu.is_vulkan or gpu.is_metal;

/// Tokenizer module (shared across all backends).
pub const tokenizer_mod = @import("../model/tokenizer.zig");
/// Forward-pass module, selected by the active backend.
pub const forward_mod = if (gpu.is_metal) @import("../compute/forward_metal.zig") else @import("../compute/forward.zig");
/// Model-loading module, selected by the active backend.
pub const loader_mod = if (gpu.is_metal) @import("../model/loader_metal.zig") else @import("../model/loader.zig");
/// Model-manager module, selected by the active backend.
pub const model_manager_mod = if (gpu.is_metal) @import("model_manager_metal.zig") else @import("model_manager.zig");

/// Backend-specific inference engine that runs the forward pass.
pub const InferenceEngine = forward_mod.InferenceEngine;
/// Per-sequence decode state (KV cache position, token history, etc.).
pub const DecodeState = forward_mod.DecodeState;
/// Loaded model handle (weights, hyperparams, GGUF metadata).
pub const Model = loader_mod.Model;
/// Manages loading, unloading, and switching between models at runtime.
pub const ModelManager = model_manager_mod.ModelManager;

/// Token sampling parameters (shared across Vulkan and Metal backends).
pub const SamplingParams = forward_mod.SamplingParams;

/// Enable logits readback from GPU so sampling can inspect raw logits.
pub fn enableLogitsReadback(_engine: *InferenceEngine) void {
    if (comptime gpu.is_vulkan) {
        _engine.enableLogitsReadback();
    }
}

/// Return whether logits readback is currently enabled on the engine.
pub fn logitsReadbackEnabled(_engine: *const InferenceEngine) bool {
    if (comptime gpu.is_vulkan) {
        return _engine.logits_readback_enabled;
    }
    // Metal uses UMA — logits are always CPU-accessible.
    return true;
}

/// Set the logits readback flag on the engine (Vulkan-only, no-op on Metal).
pub fn setLogitsReadbackEnabled(_engine: *InferenceEngine, _enabled: bool) void {
    if (comptime gpu.is_vulkan) {
        _engine.logits_readback_enabled = _enabled;
    }
}

/// Enable GPU kernel profiling on the inference engine.
pub fn enableProfiling(_engine: *InferenceEngine) !void {
    if (comptime gpu.is_vulkan) {
        try _engine.enableProfiling();
    } else if (comptime gpu.is_metal) {
        try _engine.enableProfiling();
    }
}

/// Run a single autoregressive decode step, advancing the KV cache by one token.
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

/// Sample the next token from the model's logit distribution.
pub fn sample(
    _engine: *const InferenceEngine,
    _state: *const DecodeState,
    _params: SamplingParams,
    _random: std.Random,
) u32 {
    if (comptime gpu.is_vulkan) {
        return _engine.sample(_state, _params, _random);
    }
    return _engine.sample(_state.generated_tokens.items, _params, _random);
}
