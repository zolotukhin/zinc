#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint n_used;
    uint src_stride;
};

kernel void main0(
    device float* accum [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const uint* routing [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= p.n) return;

    float sum = 0.0f;
    for (uint expert = 0u; expert < p.n_used; expert++) {
        const float weight = as_type<float>(routing[p.n_used + expert]);
        sum += weight * src[expert * p.src_stride + id];
    }
    accum[id] += sum;
}
