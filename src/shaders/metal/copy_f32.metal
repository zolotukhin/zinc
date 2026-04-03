#include <metal_stdlib>
using namespace metal;

struct CopyF32Push {
    uint n;
};

kernel void main0(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant CopyF32Push& p [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < p.n) {
        dst[id] = src[id];
    }
}
