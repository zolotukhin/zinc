#include <metal_stdlib>

using namespace metal;

struct Params {
    uint n_tokens;
    uint n_experts;
    uint k;
    uint routing_stride;
    uint ids_stride;
};

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const uint* routing [[buffer(1)]],
    device uint* counts [[buffer(2)]],
    device uint* ids [[buffer(3)]],
    uint expert_id [[thread_position_in_threadgroup]]
) {
    if (expert_id >= p.n_experts) {
        return;
    }

    uint n = 0u;
    for (uint token = 0u; token < p.n_tokens; token++) {
        device const uint* row = routing + token * p.routing_stride;
        for (uint slot = 0u; slot < p.k; slot++) {
            if (row[slot] == expert_id) {
                ids[expert_id * p.ids_stride + n] = token * p.k + slot;
                n++;
            }
        }
    }

    counts[expert_id] = n;
}
