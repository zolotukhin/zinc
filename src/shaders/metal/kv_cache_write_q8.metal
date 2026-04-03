#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint dst_offset;
    uint dst_offset_bytes;
};

inline char quantizeQ8(float value, float inv_scale) {
    if (inv_scale == 0.0f) return char(0);
    const int q = clamp(int(rint(value * inv_scale)), -127, 127);
    return char(q);
}

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* src_k [[buffer(1)]],
    device const float* src_v [[buffer(2)]],
    device uchar* dst_k [[buffer(3)]],
    device uchar* dst_v [[buffer(4)]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (block >= p.n || lane >= 32u) return;

    const uint src_base = block * 32u;
    const float k_value = src_k[src_base + lane];
    const float v_value = src_v[src_base + lane];

    const float k_abs_max = simd_max(fast::abs(k_value));
    const float v_abs_max = simd_max(fast::abs(v_value));
    const float k_scale = k_abs_max > 0.0f ? k_abs_max / 127.0f : 0.0f;
    const float v_scale = v_abs_max > 0.0f ? v_abs_max / 127.0f : 0.0f;
    const float k_inv_scale = k_scale > 0.0f ? 1.0f / k_scale : 0.0f;
    const float v_inv_scale = v_scale > 0.0f ? 1.0f / v_scale : 0.0f;

    device uchar* k_block = dst_k + p.dst_offset_bytes + block * 34u;
    device uchar* v_block = dst_v + p.dst_offset_bytes + block * 34u;

    if (lane == 0u) {
        *(device ushort*)(k_block) = as_type<ushort>(half(k_scale));
        *(device ushort*)(v_block) = as_type<ushort>(half(v_scale));
    }

    k_block[2u + lane] = as_type<uchar>(quantizeQ8(k_value, k_inv_scale));
    v_block[2u + lane] = as_type<uchar>(quantizeQ8(v_value, v_inv_scale));
}
