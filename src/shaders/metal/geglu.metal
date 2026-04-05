#include <metal_stdlib>
using namespace metal;

struct GeGLUParams {
    uint N;
};

kernel void main0(
    constant GeGLUParams& params [[buffer(0)]],
    device const float* gate      [[buffer(1)]],
    device float* y               [[buffer(2)]],
    device const float* up        [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.N) return;

    float g = gate[idx];
    // gelu_tanh(g) = 0.5 * g * (1.0 + tanh(sqrt(2/pi) * (g + 0.044715 * g^3)))
    float g3 = g * g * g;
    float inner = 0.7978845608f * (g + 0.044715f * g3);
    float gelu_g = 0.5f * g * (1.0f + tanh(inner));
    y[idx] = gelu_g * up[idx];
}
