#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct PushConstants {
    uint N;
    uint scale_bits;
};

kernel void main0(
    constant PushConstants& pc [[buffer(0)]],
    device float* data [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < pc.N) {
        data[gid] *= as_type<float>(pc.scale_bits);
    }
}
