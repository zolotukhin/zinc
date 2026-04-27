#include <metal_stdlib>
using namespace metal;

// Scatter route-slot ordered expert outputs back to token order.
//
// src is [n_tokens * k][hidden_dim] in route order.
// routing is [n_tokens][ids(k), weights(k)].
// dst is [n_tokens][hidden_dim].

struct Params {
    uint n_tokens;
    uint hidden_dim;
    uint n_experts;
    uint k;
    uint routing_stride;
    uint scale_offset;
    uint debug;
};

kernel void main0(
    constant Params& p         [[buffer(0)]],
    device const uint* routing [[buffer(1)]],
    device const float* src    [[buffer(2)]],
    device float* dst          [[buffer(3)]],
    device const float* scales [[buffer(4)]],
    uint id                   [[thread_position_in_grid]]
) {
    const uint total = p.n_tokens * p.hidden_dim;
    if (id >= total || p.hidden_dim == 0u || p.k == 0u) {
        return;
    }

    const uint token = id / p.hidden_dim;
    const uint dim = id - token * p.hidden_dim;
    device const uint* row = routing + token * p.routing_stride;

    float sum = 0.0f;
    for (uint slot = 0u; slot < p.k; slot++) {
        const uint expert_id = row[slot];
        if (p.debug != 0u && expert_id >= p.n_experts) {
            dst[id] = as_type<float>(0x7fc00000u);
            return;
        }

        const float weight = as_type<float>(row[p.k + slot]);
        sum += weight * scales[p.scale_offset + expert_id] * src[(token * p.k + slot) * p.hidden_dim + dim];
    }

    dst[id] += sum;
}
