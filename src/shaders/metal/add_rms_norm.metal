#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint n;
    float eps;
};

// Fused residual-add + RMS norm: hidden[i] += residual[i], then
// output[i] = weights[i] * rms_norm(hidden[i]).
// Eliminates one barrier and one dispatch per layer vs separate scale_acc + rms_norm.
// Adapted from llama.cpp's fused norm approach (ggml-metal-common.cpp op fusion).
kernel void main0(
    constant Params& p [[buffer(0)]],
    device float* hidden [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const float* weights [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint subgroup_size [[thread_execution_width]]
) {
    const uint width = max(subgroup_size, 1u);
    const uint base = group_id * p.n;

    // Pass 1: Add residual and compute sum of squares
    float sum_sq = 0.0f;
    for (uint i = tid; i < p.n; i += width) {
        const float v = hidden[base + i] + residual[base + i];
        hidden[base + i] = v;
        sum_sq += v * v;
    }

    const float rms_inv = rsqrt((simd_sum(sum_sq) / float(p.n)) + p.eps);

    // Pass 2: Normalize and write output
    for (uint i = tid; i < p.n; i += width) {
        output[base + i] = weights[i] * (hidden[base + i] * rms_inv);
    }
}
