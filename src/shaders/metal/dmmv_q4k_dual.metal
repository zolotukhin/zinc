#include <metal_stdlib>
using namespace metal;

// Dual Q4_K DMMV for same-input gate/up projections.
//
// This mirrors llama.cpp's merged gate_up MoE graph shape for the Gemma shared
// expert: both matrices read the same input vector, so keep that vector in
// threadgroup memory and compute the two projections in one dispatch.

struct DualQ4KDmmvPush {
    uint M0;
    uint M1;
    uint K;
    uint a0_offset;
    uint a1_offset;
    uint x_offset;
    uint y0_offset;
    uint y1_offset;
};

inline float2 get_scale_min_k4(uint j, device const uchar* sc) {
    if (j < 4) {
        return float2(float(sc[j] & 63), float(sc[j + 4] & 63));
    }
    return float2(
        float((sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4)),
        float(((sc[j + 4] >> 4) & 0x0F) | ((sc[j] >> 6) << 4))
    );
}

inline void accumulate_q4k(
    device const uchar* block,
    threadgroup float4* x_cache4,
    uint bi,
    uint lane,
    thread float& acc
) {
    const float d = float(as_type<half>(*(device const ushort*)(block)));
    const float dmin = float(as_type<half>(*(device const ushort*)(block + 2)));
    device const uchar* scales = block + 4;
    device const uchar* quants = block + 16;

    const uint byte_off = lane * 4u;
    const uint j = byte_off / 32u;
    const uint local_off = byte_off % 32u;

    const uchar4 qbytes = *(device const uchar4*)(quants + byte_off);
    const float2 sm_lo = get_scale_min_k4(j * 2u, scales);
    const float2 sm_hi = get_scale_min_k4(j * 2u + 1u, scales);

    const float d_sc_lo = d * sm_lo.x;
    const float d_m_lo = dmin * sm_lo.y;
    const float d_sc_hi = d * sm_hi.x;
    const float d_m_hi = dmin * sm_hi.y;

    const uint col_lo = bi * 256u + j * 64u + local_off;
    const uint col_hi = col_lo + 32u;

    const float4 x_lo = x_cache4[col_lo >> 2];
    const float4 x_hi = x_cache4[col_hi >> 2];

    const uchar4 q_lo = uchar4(
        qbytes.x & 0x0F,
        qbytes.y & 0x0F,
        qbytes.z & 0x0F,
        qbytes.w & 0x0F
    );
    const uchar4 q_hi = uchar4(
        qbytes.x >> 4,
        qbytes.y >> 4,
        qbytes.z >> 4,
        qbytes.w >> 4
    );

    const float4 lo_vals = fma(float4(q_lo), float4(d_sc_lo), float4(-d_m_lo));
    const float4 hi_vals = fma(float4(q_hi), float4(d_sc_hi), float4(-d_m_hi));

    acc += dot(lo_vals, x_lo) + dot(hi_vals, x_hi);
}

#define TG_SIZE 256
#define MAX_K_VEC4 1344
#define ROWS_PER_SIMDGROUP 2
#define ROWS_PER_TG ((TG_SIZE / 32) * ROWS_PER_SIMDGROUP)

kernel void main0(
    device const uchar* W0                        [[buffer(0)]],
    device const uchar* W1                        [[buffer(1)]],
    constant DualQ4KDmmvPush& p                   [[buffer(2)]],
    device const float* X                         [[buffer(3)]],
    device float* Y0                              [[buffer(4)]],
    device float* Y1                              [[buffer(5)]],
    uint3 tg_pos                                  [[threadgroup_position_in_grid]],
    uint local_id                                 [[thread_index_in_threadgroup]],
    uint lane                                     [[thread_index_in_simdgroup]],
    uint sg_idx                                   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float4 x_cache4[MAX_K_VEC4];

    device const float* input = X + (p.x_offset >> 2);
    const uint k_vec4 = p.K >> 2;
    for (uint i = local_id; i < k_vec4; i += TG_SIZE) {
        x_cache4[i] = *(device const float4*)(input + (i << 2));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row0 = tg_pos.x * ROWS_PER_TG + sg_idx * ROWS_PER_SIMDGROUP;
    const uint row1 = row0 + 1u;
    const bool valid0 = row0 < p.M0;
    const bool valid1 = row1 < p.M0;
    if (!valid0 && !valid1) {
        return;
    }

    const uint bpr = p.K / 256u;
    const ulong row_bytes = ulong(bpr) * 144ul;

    device const uchar* w0_row0 = W0 + ulong(p.a0_offset) + ulong(row0) * row_bytes;
    device const uchar* w0_row1 = W0 + ulong(p.a0_offset) + ulong(row1) * row_bytes;
    device const uchar* w1_row0 = W1 + ulong(p.a1_offset) + ulong(row0) * row_bytes;
    device const uchar* w1_row1 = W1 + ulong(p.a1_offset) + ulong(row1) * row_bytes;

    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    for (uint bi = 0u; bi < bpr; bi++) {
        if (valid0) {
            accumulate_q4k(w0_row0 + ulong(bi) * 144ul, x_cache4, bi, lane, acc00);
            accumulate_q4k(w1_row0 + ulong(bi) * 144ul, x_cache4, bi, lane, acc10);
        }
        if (valid1) {
            accumulate_q4k(w0_row1 + ulong(bi) * 144ul, x_cache4, bi, lane, acc01);
            accumulate_q4k(w1_row1 + ulong(bi) * 144ul, x_cache4, bi, lane, acc11);
        }
    }

    const float sum00 = simd_sum(acc00);
    const float sum01 = simd_sum(acc01);
    const float sum10 = simd_sum(acc10);
    const float sum11 = simd_sum(acc11);

    if (lane == 0u) {
        device float* out0 = Y0 + (p.y0_offset >> 2);
        device float* out1 = Y1 + (p.y1_offset >> 2);
        if (valid0) {
            out0[row0] = sum00;
            out1[row0] = sum10;
        }
        if (valid1) {
            out0[row1] = sum01;
            out1[row1] = sum11;
        }
    }
}
