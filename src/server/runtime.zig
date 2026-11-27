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
/// Model-native MTP needs a verifier and rollback path in the serving loop.
/// Retaining NextN weights alone does not make the HTTP runtime capable of MTP.
pub const supports_mtp = false;
pub const mtp_unavailable_reason = if (gpu.is_vulkan)
    "Model-native MTP is not implemented by the Vulkan chat runtime."
else if (gpu.is_metal)
    "Model-native MTP is not implemented by the Metal chat runtime."
else
    "Model-native MTP is not implemented by this HTTP runtime.";

pub fn mtpRequestError(enabled: ?bool) ?[]const u8 {
    return if ((enabled orelse false) and !supports_mtp) mtp_unavailable_reason else null;
}

/// Tokenizer module (shared across all backends).
pub const tokenizer_mod = @import("../model/tokenizer.zig");
/// Forward-pass module, selected by the active backend.
pub const forward_mod = if (gpu.is_metal) @import("../compute/forward_metal.zig") else @import("../compute/vulkan_inference.zig");
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
/// On Metal (UMA) logits are always CPU-accessible, so this is a no-op.
/// @param _engine Inference engine whose readback mode is updated.
pub fn enableLogitsReadback(_engine: *InferenceEngine) void {
    if (comptime gpu.is_vulkan) {
        _engine.enableLogitsReadback();
    } else if (comptime gpu.is_metal) {
        _engine.enableLogitsReadback();
    }
}

/// Return whether logits readback is currently enabled on the engine.
/// Always returns `true` on Metal because UMA makes logits CPU-accessible without an explicit readback step.
/// @param _engine Inference engine to query.
/// @returns `true` if the engine will expose logits after each decode step.
pub fn logitsReadbackEnabled(_engine: *const InferenceEngine) bool {
    if (comptime gpu.is_vulkan) {
        return _engine.logitsReadbackEnabled();
    } else if (comptime gpu.is_metal) {
        return _engine.logits_readback_enabled;
    }
    return false;
}

/// Set the logits readback intent flag on backends that can elide full logit materialization.
pub fn setLogitsReadbackEnabled(_engine: *InferenceEngine, _enabled: bool) void {
    if (comptime gpu.is_vulkan) {
        _engine.setLogitsReadbackEnabled(_enabled);
    } else if (comptime gpu.is_metal) {
        _engine.logits_readback_enabled = _enabled;
    }
}

/// Enable GPU kernel profiling on the inference engine.
/// Supported on both Vulkan and Metal backends; calls the backend's own `enableProfiling` method.
/// @param _engine Inference engine to configure for profiling.
/// @returns An error if the backend's profiling setup fails (e.g. out of GPU resources).
pub fn enableProfiling(_engine: *InferenceEngine) !void {
    if (comptime gpu.is_vulkan) {
        try _engine.enableProfiling();
    } else if (comptime gpu.is_metal) {
        try _engine.enableProfiling();
    }
}

/// Run a single autoregressive decode step, advancing the KV cache by one token.
/// @param _engine Inference engine that owns the model weights and KV cache.
/// @param _state Per-sequence decode state tracking position and token history.
/// @param _token_id Input token to feed into the model for this step.
/// @param _collect_output When `true`, copy output logits to CPU (Vulkan only; ignored on Metal where logits are always accessible).
/// @returns An error if the GPU submission or synchronisation fails.
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
/// @param _engine Inference engine holding the current logits.
/// @param _state Decode state used to retrieve generated-token history for repetition penalty.
/// @param _params Sampling configuration (temperature, top-p, top-k, repetition penalty, etc.).
/// @param _random PRNG source for stochastic sampling.
/// @returns The sampled token ID.
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
