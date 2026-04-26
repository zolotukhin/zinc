#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint n_used;
    uint src_stride;
    uint scale_offset;
};

kernel void main0(
    device float* accum [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const uint* routing [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    device const float* expert_scales [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= p.n) return;

    float sum = 0.0f;
    for (uint slot = 0u; slot < p.n_used; slot++) {
        const uint expert_id = routing[slot];
        const float weight = as_type<float>(routing[p.n_used + slot]);
        sum += weight * expert_scales[p.scale_offset + expert_id] * src[slot * p.src_stride + id];
    }
    accum[id] += sum;
}
