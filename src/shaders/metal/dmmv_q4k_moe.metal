#include <metal_stdlib>
using namespace metal;

struct MoeDmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint expert_stride;
    uint x_expert_stride;
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

#define TG_SIZE 256
#define ROWS_PER_SIMDGROUP 2
#define ROWS_PER_TG ((TG_SIZE / 32) * ROWS_PER_SIMDGROUP)

kernel void main0(
    device const uchar* W [[buffer(0)]],
    constant MoeDmmvPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    device const uint* expert_ids [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 local_pos [[thread_position_in_threadgroup]]
) {
    const uint expert_slot = tg_pos.y;
    const uint expert_id = expert_ids[expert_slot];
    device const float* input = X + (p.x_offset / 4) + expert_slot * p.x_expert_stride;
    threadgroup float4 x_cache4[1024];

    const uint local_id = local_pos.x;
    const uint sg_idx = local_id / 32;
    const uint lane = local_id % 32;

    const uint k_vec4 = p.K >> 2;
    for (uint i = local_id; i < k_vec4; i += TG_SIZE) {
        x_cache4[i] = *(device const float4*)(input + (i << 2));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row0 = tg_pos.x * ROWS_PER_TG + sg_idx * ROWS_PER_SIMDGROUP;
    uint row1 = row0 + 1;
    const bool valid0 = row0 < p.M;
    const bool valid1 = row1 < p.M;
    if (!valid0 && !valid1) return;

    uint bpr = p.K / 256;
    ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);
    const ulong row_bytes = ulong(bpr) * 144;
    device const uchar* row_ptr0 = W + expert_base + ulong(row0) * row_bytes;
    device const uchar* row_ptr1 = W + expert_base + ulong(row1) * row_bytes;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint bi = 0; bi < bpr; bi++) {
        device const uchar* block0 = row_ptr0 + bi * 144;

        float d0 = float(as_type<half>(*(device const ushort*)(block0)));
        float dmin0 = float(as_type<half>(*(device const ushort*)(block0 + 2)));

        device const uchar* scales0 = block0 + 4;
        device const uchar* quants0 = block0 + 16;

        uint byte_off = lane * 4;
        uint j = byte_off / 32;
        uint local_off = byte_off % 32;

        uchar4 qbytes0 = *(device const uchar4*)(quants0 + byte_off);

        float2 sm_lo0 = get_scale_min_k4(j * 2, scales0);
        float2 sm_hi0 = get_scale_min_k4(j * 2 + 1, scales0);

        float d_sc_lo0 = d0 * sm_lo0.x;
        float d_m_lo0 = dmin0 * sm_lo0.y;
        float d_sc_hi0 = d0 * sm_hi0.x;
        float d_m_hi0 = dmin0 * sm_hi0.y;

        uint col_lo = bi * 256 + j * 64 + local_off;
        uint col_hi = col_lo + 32;

        float4 x_lo = x_cache4[col_lo >> 2];
        float4 x_hi = x_cache4[col_hi >> 2];

        uchar4 q_lo0 = uchar4(
            qbytes0.x & 0x0F,
            qbytes0.y & 0x0F,
            qbytes0.z & 0x0F,
            qbytes0.w & 0x0F
        );
        uchar4 q_hi0 = uchar4(
            qbytes0.x >> 4,
            qbytes0.y >> 4,
            qbytes0.z >> 4,
            qbytes0.w >> 4
        );

        float4 lo_vals0 = fma(float4(q_lo0), float4(d_sc_lo0), float4(-d_m_lo0));
        float4 hi_vals0 = fma(float4(q_hi0), float4(d_sc_hi0), float4(-d_m_hi0));

        acc0 += dot(lo_vals0, x_lo);
        acc0 += dot(hi_vals0, x_hi);

        if (valid1) {
            device const uchar* block1 = row_ptr1 + bi * 144;
            float d1 = float(as_type<half>(*(device const ushort*)(block1)));
            float dmin1 = float(as_type<half>(*(device const ushort*)(block1 + 2)));
            device const uchar* scales1 = block1 + 4;
            device const uchar* quants1 = block1 + 16;

            uchar4 qbytes1 = *(device const uchar4*)(quants1 + byte_off);
            float2 sm_lo1 = get_scale_min_k4(j * 2, scales1);
            float2 sm_hi1 = get_scale_min_k4(j * 2 + 1, scales1);

            float d_sc_lo1 = d1 * sm_lo1.x;
            float d_m_lo1 = dmin1 * sm_lo1.y;
            float d_sc_hi1 = d1 * sm_hi1.x;
            float d_m_hi1 = dmin1 * sm_hi1.y;

            uchar4 q_lo1 = uchar4(
                qbytes1.x & 0x0F,
                qbytes1.y & 0x0F,
                qbytes1.z & 0x0F,
                qbytes1.w & 0x0F
            );
            uchar4 q_hi1 = uchar4(
                qbytes1.x >> 4,
                qbytes1.y >> 4,
                qbytes1.z >> 4,
                qbytes1.w >> 4
            );

            float4 lo_vals1 = fma(float4(q_lo1), float4(d_sc_lo1), float4(-d_m_lo1));
            float4 hi_vals1 = fma(float4(q_hi1), float4(d_sc_hi1), float4(-d_m_hi1));

            acc1 += dot(lo_vals1, x_lo);
            acc1 += dot(hi_vals1, x_hi);
        }
    }

    float sum0 = simd_sum(acc0);
    float sum1 = simd_sum(acc1);
    if (lane == 0) {
        device float* out = Y + (p.y_offset / 4) + expert_slot * p.M;
        if (valid0) out[row0] = sum0;
        if (valid1) out[row1] = sum1;
    }
}
