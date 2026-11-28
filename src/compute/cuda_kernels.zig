//! Kernel source shared by accelerator forwards because the context caches one module.
//! @section CUDA Runtime
//!
//! The HIP compiler resolves both the common kernels and Inkling's text
//! kernels from this module, including after a server model switch.
const std = @import("std");
const is_rocm = std.mem.eql(u8, @import("build_options").backend, "rocm");

pub const source = @embedFile("../shaders/cuda/kernels.cu") ++
    if (is_rocm) "\n" ++ @embedFile("../shaders/cuda/inkling.cu") ++ "\n" ++ @embedFile("../shaders/cuda/inkling_dspark.cu") else "";
