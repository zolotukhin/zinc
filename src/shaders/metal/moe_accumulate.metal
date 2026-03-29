#include <metal_stdlib>
using namespace metal;

// Fused MoE weighted accumulate: dst[i] += sum(w[j] * src_j[i]) for all experts + shared expert.
// Replaces 8+1 sequential scale_accumulate dispatches with barriers (320 pipeline flushes per token).
//
// Push constants (buffer 10):
//   n:     number of elements (hidden_dim)
//   w0-w7: expert weights (softmax-normalized)
//   w_sh:  shared expert weight (sigmoid gate value, 0 if no shared expert)

struct Params {
    uint n;
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
    device float* dst        [[buffer(0)]],
    device const float* e0   [[buffer(1)]],
    device const float* e1   [[buffer(2)]],
    device const float* e2   [[buffer(3)]],
    device const float* e3   [[buffer(4)]],
    device const float* e4   [[buffer(5)]],
    device const float* e5   [[buffer(6)]],
    device const float* e6   [[buffer(7)]],
    device const float* e7   [[buffer(8)]],
    device const float* sh   [[buffer(9)]],
    constant Params& p       [[buffer(10)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= p.n) return;
    dst[id] += p.w0 * e0[id] + p.w1 * e1[id] + p.w2 * e2[id] + p.w3 * e3[id]
             + p.w4 * e4[id] + p.w5 * e5[id] + p.w6 * e6[id] + p.w7 * e7[id]
             + p.w_sh * sh[id];
}
