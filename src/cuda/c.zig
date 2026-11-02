//! Shared C import for the native CUDA/ROCm shim. All native backend modules
//! import from here to keep translated handle types identical across units.
//! @section CUDA Runtime
//!
//! Keeping the translated bindings in one place avoids duplicate opaque C types across
//! Zig compilation units, which is critical for safely passing shim handles
//! between the shared device, buffer, pipeline, and command helpers. The selected
//! build imports the NVIDIA CUDA or AMD HIP implementation of the same C ABI.
/// Raw CUDA shim C bindings (Driver API + NVRTC) imported from cuda_shim.h.
pub const shim = @import("cuda_c");
