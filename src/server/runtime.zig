//! Backend-specific server runtime aliases.
//! Keeps the HTTP/routes layer shared across Vulkan and Metal backends.
const gpu = @import("../gpu/interface.zig");

pub const tokenizer_mod = @import("../model/tokenizer.zig");
pub const forward_mod = if (gpu.is_metal) @import("../compute/forward_metal.zig") else @import("../compute/forward.zig");
pub const loader_mod = if (gpu.is_metal) @import("../model/loader_metal.zig") else @import("../model/loader.zig");

pub const InferenceEngine = forward_mod.InferenceEngine;
pub const DecodeState = forward_mod.DecodeState;
pub const Model = loader_mod.Model;
