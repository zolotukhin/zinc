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
    // Clamp + precise::tanh mirror the non-batched geglu.metal, which prevents
    // the fast tanh from producing NaN on M4 when |inner| is large enough that
    // exp overflows f32 (tanh saturates to ±1 well before |x|=15).
    float inner = 0.7978845608f * (g + 0.044715f * g3);
    inner = clamp(inner, -15.0f, 15.0f);
    const float gelu_g = 0.5f * g * (1.0f + precise::tanh(inner));
    out[idx] = gelu_g * up[idx];
}
