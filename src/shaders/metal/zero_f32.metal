#include <metal_stdlib>
using namespace metal;

struct ZeroF32Push {
    uint n;
};

kernel void main0(
    device float* dst [[buffer(0)]],
    constant ZeroF32Push& p [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < p.n) {
        dst[id] = 0.0f;
    }
}
