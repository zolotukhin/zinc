//! Vulkan C bindings — shared import for all vulkan/ modules.
/// Raw Vulkan C API bindings imported from the system vulkan.h header.
pub const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
