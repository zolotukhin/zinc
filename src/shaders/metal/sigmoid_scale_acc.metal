#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint unused_scale_bits;
};

kernel void main0(
    device float* accum [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const float* gate [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= p.n) return;
    const float gate_val = 1.0f / (1.0f + exp(-gate[0]));
    accum[id] += gate_val * src[id];
}
