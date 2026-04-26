#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    float scale;
};

kernel void main0(
    device float* hidden [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= p.n) return;
    hidden[id] = (hidden[id] + residual[id]) * p.scale;
}
