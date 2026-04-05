//! Backend-selected model manager for the HTTP server.
const gpu = @import("../gpu/interface.zig");

const impl = if (gpu.is_metal)
    @import("model_manager_metal.zig")
else
    @import("model_manager.zig");

/// Specification describing which model to load (path, managed ID, or default).
pub const LoadSpec = impl.LoadSpec;
/// Owns the loaded model, tokenizer, and inference engine for the HTTP server.
pub const ModelManager = impl.ModelManager;
