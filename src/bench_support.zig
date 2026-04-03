//! Re-exports for the Metal hot-decode benchmark binary.
//! Provides a single import surface so `bench_hot_decode.zig` can reach
//! Metal device, model loader, tokenizer, and forward-pass modules.
//! @section Inference Runtime

/// Metal device initialisation and capability detection.
pub const metal_device = @import("metal/device.zig");
/// Metal-specific GGUF model loader (zero-copy mmap).
pub const metal_loader = @import("model/loader_metal.zig");
/// BPE tokenizer, chat templates, and thinking-toggle support.
pub const tokenizer_mod = @import("model/tokenizer.zig");
/// Metal inference engine: prefill and decode loops.
pub const forward_metal = @import("compute/forward_metal.zig");
