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
    const float g = gate[idx];
    const float g3 = g * g * g;
    const float inner = 0.7978845608f * (g + 0.044715f * g3);
    out[idx] = (0.5f * g * (1.0f + tanh(inner))) * up[idx];
}
