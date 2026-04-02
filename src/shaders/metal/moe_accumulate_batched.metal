#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint expert_stride;
    float w0;
    float w1;
    float w2;
    float w3;
    float w4;
    float w5;
    float w6;
    float w7;
    float w_sh;
};

kernel void main0(
    device float* dst [[buffer(0)]],
    device const float* experts [[buffer(1)]],
    device const float* sh [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= p.n) return;

    const uint s = p.expert_stride;
    float sum = p.w0 * experts[id];
    sum += p.w1 * experts[s + id];
    sum += p.w2 * experts[2 * s + id];
    sum += p.w3 * experts[3 * s + id];
    sum += p.w4 * experts[4 * s + id];
    sum += p.w5 * experts[5 * s + id];
    sum += p.w6 * experts[6 * s + id];
    sum += p.w7 * experts[7 * s + id];
    sum += p.w_sh * sh[id];

    dst[id] += sum;
}
