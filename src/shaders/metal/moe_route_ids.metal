#include <metal_stdlib>
using namespace metal;

// Flatten batched top-k routing into route-slot expert IDs.
//
// routing is [n_tokens][ids(k), weights(k)] as written by softmax_topk_batched.
// ids is [n_tokens * k] in route order: route = token * k + slot.

struct Params {
    uint n_tokens;
    uint k;
    uint routing_stride;
};

kernel void main0(
    constant Params& p         [[buffer(0)]],
    device const uint* routing [[buffer(1)]],
    device uint* ids           [[buffer(2)]],
    uint route                 [[thread_position_in_grid]]
) {
    const uint total = p.n_tokens * p.k;
    if (route >= total || p.k == 0u) {
        return;
    }

    const uint token = route / p.k;
    const uint slot = route - token * p.k;
    ids[route] = routing[token * p.routing_stride + slot];
}
