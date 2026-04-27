#include <metal_stdlib>
using namespace metal;

// Batched shared-expert gate accumulation.
//
// accum/src are [n_tokens][hidden_dim], gate is [n_tokens][1].
// For each token: accum += sigmoid(gate[token]) * src.

struct Params {
    uint n_tokens;
    uint hidden_dim;
};

kernel void main0(
    device float* accum [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const float* gate [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    const uint total = p.n_tokens * p.hidden_dim;
    if (id >= total || p.hidden_dim == 0u) {
        return;
    }

    const uint token = id / p.hidden_dim;
    const float gate_val = 1.0f / (1.0f + exp(-gate[token]));
    accum[id] += gate_val * src[id];
}
