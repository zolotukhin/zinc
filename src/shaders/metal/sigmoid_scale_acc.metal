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
    uint id [[thread_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (id >= p.n) return;
    threadgroup float gate_val;
    if (lane == 0u) {
        gate_val = 1.0f / (1.0f + exp(-gate[0]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    accum[id] += gate_val * src[id];
}
