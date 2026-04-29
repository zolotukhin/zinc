#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint use_gate;
};

// Fused tail of the Gemma MoE post-projection sequence:
//   moe_out += [sigmoid(gate[0]) *] down       (combine routed + shared)
//   hidden  += moe_out                         (residual to next layer)
//
// Effective:  hidden[i] += moe_out[i] + scale_b * down[i]
// where scale_b = 1.0 when use_gate==0, else sigmoid(gate[0]).
//
// Replaces sigmoid_scale_acc/scale_accumulate + barrier + scale_accumulate
// when no separate post_ffn_norm sits between the two adds (true for
// Gemma 4, which only has the per-branch post_ffw_norm_1/2 already
// applied earlier in the layer). Each thread reads the 1-element gate
// directly so no threadgroup synchronization is required.
kernel void main0(
    device float* hidden [[buffer(0)]],
    device const float* moe_out [[buffer(1)]],
    device const float* down [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= p.n) return;
    float scale_b = 1.0f;
    if (p.use_gate != 0u) {
        scale_b = 1.0f / (1.0f + exp(-gate[0]));
    }
    hidden[id] += moe_out[id] + scale_b * down[id];
}
