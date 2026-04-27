#include <metal_stdlib>
using namespace metal;

// Gather token-ordered MoE inputs into route-slot order.
//
// `src` is [n_tokens][hidden_dim].
// `dst` is [n_tokens * k][hidden_dim], indexed by route = token * k + slot.
// The grouped expert DMMV kernels consume this route-slot layout after
// moe_route_pack has grouped route ids by expert.

struct Params {
    uint n_tokens;
    uint hidden_dim;
    uint n_experts;
    uint k;
    uint routing_stride;
    uint debug;
};

kernel void main0(
    constant Params& p         [[buffer(0)]],
    device const uint* routing [[buffer(1)]],
    device const float* src    [[buffer(2)]],
    device float* dst          [[buffer(3)]],
    uint id                    [[thread_position_in_grid]]
) {
    const uint route_slots = p.n_tokens * p.k;
    const uint total = route_slots * p.hidden_dim;
    if (id >= total || p.hidden_dim == 0u || p.k == 0u) {
        return;
    }

    const uint route = id / p.hidden_dim;
    const uint dim = id - route * p.hidden_dim;
    const uint token = route / p.k;
    const uint slot = route - token * p.k;

    bool invalid = false;
    if (p.debug != 0u) {
        if (p.routing_stride < p.k * 2u || token >= p.n_tokens || slot >= p.k) {
            invalid = true;
        } else {
            const uint expert_id = routing[token * p.routing_stride + slot];
            if (expert_id >= p.n_experts) {
                invalid = true;
            }
        }
    }

    if (invalid) {
        dst[id] = as_type<float>(0x7fc00000u);
        return;
    }

    dst[id] = src[token * p.hidden_dim + dim];
}
