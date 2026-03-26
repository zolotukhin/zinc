// Vulkan C bindings — shared import for all vulkan/ modules.
pub const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
