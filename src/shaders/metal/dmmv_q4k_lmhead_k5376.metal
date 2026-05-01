#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
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

// Large-M Q4_K vocab projection for Gemma 31B (K=5376).
// This keeps llama.cpp's single-token matvec dispatch model from
// `ggml_metal_op_mul_mat`, but specializes the very large row count to reuse
// the hidden vector across 16 rows per threadgroup. Do not use this for normal
// layer projections; the 21 KiB threadgroup input cache only amortizes on the
// LM head's 262k-row shape.
#define TG_SIZE 512
#define ROWS_PER_TG (TG_SIZE / 32)
#define MAX_K_VEC4 1536

kernel void main0(
    device const uchar* W [[buffer(0)]],
    constant DmmvPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    device const float* input = X + (p.x_offset / 4);
    threadgroup float4 x_cache4[MAX_K_VEC4];

    const uint k_vec4 = p.K >> 2;
    for (uint i = local_id; i < k_vec4; i += TG_SIZE) {
        x_cache4[i] = *(device const float4*)(input + (i << 2));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = tg_id * ROWS_PER_TG + sg_idx;
    if (row >= p.M) return;

    uint bpr = p.K / 256;
    device const uchar* row_ptr = W + p.a_offset + ulong(row) * ulong(bpr) * 144;

    float acc = 0.0f;

    for (uint bi = 0; bi < bpr; bi++) {
        device const uchar* block = row_ptr + bi * 144;

        float d = float(as_type<half>(*(device const ushort*)(block)));
        float dmin = float(as_type<half>(*(device const ushort*)(block + 2)));

        device const uchar* scales = block + 4;
        device const uchar* quants = block + 16;

        uint byte_off = lane * 4;
        uint j = byte_off / 32;
        uint local_off = byte_off % 32;

        uchar4 qbytes = *(device const uchar4*)(quants + byte_off);

        float2 sm_lo = get_scale_min_k4(j * 2, scales);
        float2 sm_hi = get_scale_min_k4(j * 2 + 1, scales);

        float d_sc_lo = d * sm_lo.x;
        float d_m_lo = dmin * sm_lo.y;
        float d_sc_hi = d * sm_hi.x;
        float d_m_hi = dmin * sm_hi.y;

        uint col_lo = bi * 256 + j * 64 + local_off;
        uint col_hi = col_lo + 32;

        float4 x_lo = x_cache4[col_lo >> 2];
        float4 x_hi = x_cache4[col_hi >> 2];

        uchar4 q_lo = uchar4(
            qbytes.x & 0x0F,
            qbytes.y & 0x0F,
            qbytes.z & 0x0F,
            qbytes.w & 0x0F
        );
        uchar4 q_hi = uchar4(
            qbytes.x >> 4,
            qbytes.y >> 4,
            qbytes.z >> 4,
            qbytes.w >> 4
        );

        float4 lo_vals = fma(float4(q_lo), float4(d_sc_lo), float4(-d_m_lo));
        float4 hi_vals = fma(float4(q_hi), float4(d_sc_hi), float4(-d_m_hi));

        acc += dot(lo_vals, x_lo);
        acc += dot(hi_vals, x_hi);
    }

    float sum = simd_sum(acc);
    if (lane == 0) {
        Y[p.y_offset / 4 + row] = sum;
    }
}
