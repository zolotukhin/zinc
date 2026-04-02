//! Backend-selected model manager for the HTTP server.
const gpu = @import("../gpu/interface.zig");

const impl = if (gpu.is_metal)
    @import("model_manager_metal.zig")
else
    @import("model_manager.zig");

pub const LoadSpec = impl.LoadSpec;
pub const ModelManager = impl.ModelManager;
