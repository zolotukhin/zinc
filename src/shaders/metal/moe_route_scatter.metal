#include <metal_stdlib>
using namespace metal;

// Scatter grouped MoE route outputs back to token order.
//
// `src` is indexed by route slot, where route = token * k + topk_slot.
// `routing` stores one row per token: [k expert ids][k f32 weights as u32].
// In debug mode the counts/packed id table is checked before accumulating.

struct Params {
    uint n_tokens;
    uint hidden_dim;
    uint n_experts;
    uint k;
    uint routing_stride;
    uint ids_stride;
    uint debug;
};

kernel void main0(
    constant Params& p            [[buffer(0)]],
    device const uint* counts     [[buffer(1)]],
    device const uint* packed_ids [[buffer(2)]],
    device const uint* routing    [[buffer(3)]],
    device const float* src       [[buffer(4)]],
    device float* dst             [[buffer(5)]],
    uint id                       [[thread_position_in_grid]]
) {
    const uint total = p.n_tokens * p.hidden_dim;
    if (id >= total || p.hidden_dim == 0u) {
        return;
    }

    const uint token = id / p.hidden_dim;
    const uint dim = id - token * p.hidden_dim;
    const uint route_slots = p.n_tokens * p.k;
    bool invalid = false;

    if (p.debug != 0u) {
        if (p.routing_stride < p.k * 2u) {
            invalid = true;
        }
        for (uint expert = 0u; expert < p.n_experts && !invalid; expert++) {
            const uint count = counts[expert];
            if (count > p.ids_stride) {
                invalid = true;
                break;
            }
            device const uint* expert_ids = packed_ids + expert * p.ids_stride;
            for (uint i = 0u; i < count; i++) {
                if (expert_ids[i] >= route_slots) {
                    invalid = true;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    device const uint* route_row = routing + token * p.routing_stride;
    for (uint slot = 0u; slot < p.k; slot++) {
        const uint expert_id = route_row[slot];
        const uint route_id = token * p.k + slot;

        if (p.debug != 0u) {
            if (expert_id >= p.n_experts) {
                invalid = true;
            } else {
                const uint count = counts[expert_id];
                bool found = false;
                if (count <= p.ids_stride) {
                    device const uint* expert_ids = packed_ids + expert_id * p.ids_stride;
                    for (uint i = 0u; i < count; i++) {
                        if (expert_ids[i] == route_id) {
                            found = true;
                            break;
                        }
                    }
                }
                if (!found) {
                    invalid = true;
                }
            }
        }

        const float weight = as_type<float>(route_row[p.k + slot]);
        sum += weight * src[route_id * p.hidden_dim + dim];
    }

    if (invalid) {
        dst[id] = as_type<float>(0x7fc00000u);
        return;
    }

    dst[id] += sum;
}
