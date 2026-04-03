#include <metal_stdlib>
using namespace metal;

struct CopyU32Push {
    uint n_words;
    uint src_offset_words;
    uint dst_offset_words;
};

kernel void main0(
    device const uint* src [[buffer(0)]],
    device uint* dst [[buffer(1)]],
    constant CopyU32Push& p [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < p.n_words) {
        dst[p.dst_offset_words + id] = src[p.src_offset_words + id];
    }
}
