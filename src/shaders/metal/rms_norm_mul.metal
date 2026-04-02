#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint n;
    float eps;
};

// Keep each threadgroup to a single simdgroup. On Apple GPUs this avoids the
// extra threadgroup-memory reduction and barriers from the SPIRV-Cross version.
kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint subgroup_size [[thread_execution_width]]
) {
    const uint width = max(subgroup_size, 1u);
    const uint base = group_id * p.n;

    float sum_sq = 0.0f;
    for (uint i = tid; i < p.n; i += width) {
        const float v = input[base + i];
        sum_sq += v * v;
    }

    const float rms_inv = rsqrt((simd_sum(sum_sq) / float(p.n)) + p.eps);

    for (uint i = tid; i < p.n; i += width) {
        output[base + i] = weights[i] * (input[base + i] * rms_inv);
    }
}
