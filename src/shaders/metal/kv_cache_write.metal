#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint dst_offset;
};

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* src_k [[buffer(1)]],
    device const float* src_v [[buffer(2)]],
    device float* dst_k [[buffer(3)]],
    device float* dst_v [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= p.n) return;

    const uint dst = p.dst_offset + id;
    dst_k[dst] = src_k[id];
    dst_v[dst] = src_v[id];
}
