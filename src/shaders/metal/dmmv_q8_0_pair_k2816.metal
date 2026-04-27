#include <metal_stdlib>
using namespace metal;

struct DualQ8DmmvPush {
    uint M0;
    uint M1;
    uint K;
    uint a0_offset;
    uint a1_offset;
    uint x_offset;
    uint y0_offset;
    uint y1_offset;
};

// Paired equal-shape Q8_0 DMMV specialized for Gemma's K=2816 projections.
// This covers the hot shared gate/up and attention Q/gate, K/V pairs on M4.
kernel void main0(
    constant DualQ8DmmvPush& p [[buffer(0)]],
    device const uchar* W0 [[buffer(1)]],
    device const uchar* W1 [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* Y0 [[buffer(4)]],
    device float* Y1 [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    const uint base_row = (tg_id * simdgroups_per_tg + sg_idx) * 2u;
    if (base_row >= p.M0 || base_row >= p.M1) return;

    device const float* input = X + (p.x_offset >> 2);
    device float* output0 = Y0 + (p.y0_offset >> 2);
    device float* output1 = Y1 + (p.y1_offset >> 2);

    constexpr uint blocks_per_row = 88u;
    constexpr ulong row_bytes = 2992ull;
    const bool has_next = base_row + 1u < p.M0 && base_row + 1u < p.M1;
    device const uchar* row00 = W0 + p.a0_offset + ulong(base_row) * row_bytes;
    device const uchar* row01 = has_next ? (row00 + row_bytes) : row00;
    device const uchar* row10 = W1 + p.a1_offset + ulong(base_row) * row_bytes;
    device const uchar* row11 = has_next ? (row10 + row_bytes) : row10;

    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    #pragma unroll
    for (uint bi = lane; bi < blocks_per_row; bi += 32u) {
        device const uchar* blk00 = row00 + bi * 34u;
        device const uchar* blk01 = row01 + bi * 34u;
        device const uchar* blk10 = row10 + bi * 34u;
        device const uchar* blk11 = row11 + bi * 34u;
        const float s00 = float(as_type<half>(*(device const ushort*)(blk00)));
        const float s01 = float(as_type<half>(*(device const ushort*)(blk01)));
        const float s10 = float(as_type<half>(*(device const ushort*)(blk10)));
        const float s11 = float(as_type<half>(*(device const ushort*)(blk11)));
        device const packed_char4* q00 = (device const packed_char4*)(blk00 + 2u);
        device const packed_char4* q01 = (device const packed_char4*)(blk01 + 2u);
        device const packed_char4* q10 = (device const packed_char4*)(blk10 + 2u);
        device const packed_char4* q11 = (device const packed_char4*)(blk11 + 2u);
        const uint x_base = bi << 5;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const float4 x = *(device const float4*)(input + x_base + (vi << 2));
            acc00 = fma(s00, dot(float4(char4(q00[vi])), x), acc00);
            acc01 = fma(s01, dot(float4(char4(q01[vi])), x), acc01);
            acc10 = fma(s10, dot(float4(char4(q10[vi])), x), acc10);
            acc11 = fma(s11, dot(float4(char4(q11[vi])), x), acc11);
        }
    }

    const float sum00 = simd_sum(acc00);
    const float sum10 = simd_sum(acc10);
    if (lane == 0u) {
        output0[base_row] = sum00;
        output1[base_row] = sum10;
    }

    if (has_next) {
        const float sum01 = simd_sum(acc01);
        const float sum11 = simd_sum(acc11);
        if (lane == 0u) {
            output0[base_row + 1u] = sum01;
            output1[base_row + 1u] = sum11;
        }
    }
}
