#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
};

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const float* up [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= p.n) return;
    const uint idx = gid.y * p.n + gid.x;
    const float x = gate[idx];
    out[idx] = (x / (1.0f + exp(-x))) * up[idx];
}
