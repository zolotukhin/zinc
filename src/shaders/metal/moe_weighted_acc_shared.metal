#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint n_used;
    uint src_stride;
    uint has_gate;
};

// Fused MoE weighted accumulate + shared expert contribution.
// Eliminates one barrier per layer vs separate moe_weighted_acc + sigmoid_scale_acc.
// Adapted from llama.cpp's fused operation approach (ggml-metal.m, use fusion = true).
//
// Computes: accum[id] += sum(w_i * expert_i[id]) + sh_weight * shared[id]
// where sh_weight = sigmoid(gate[0]) if has_gate, else 1.0.
kernel void main0(
    device float* accum [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const uint* routing [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    device const float* shared_src [[buffer(4)]],
    device const float* gate [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= p.n) return;

    float sum = 0.0f;
    for (uint expert = 0u; expert < p.n_used; expert++) {
        const float weight = as_type<float>(routing[p.n_used + expert]);
        sum += weight * src[expert * p.src_stride + id];
    }

    // Shared expert: sigmoid gating or pass-through
    const float sh_weight = (p.has_gate != 0u)
        ? (1.0f / (1.0f + exp(-gate[0])))
        : 1.0f;
    sum += sh_weight * shared_src[id];

    accum[id] += sum;
}
