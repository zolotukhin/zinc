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

// Large-M batched MoE specialization for K <= 2048. This is intentionally
// used only for the down projection path where M is the hidden size, so 32
// rows per threadgroup amortize the staged expert input vector across more rows
// without perturbing the smaller gate/up projections.
#define TG_SIZE 1024
#define ROWS_PER_TG (TG_SIZE / 32)
#define MAX_K_VEC4 512

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
    threadgroup float4 x_cache4[MAX_K_VEC4];

    const uint local_id = local_pos.x;
    const uint sg_idx = local_id / 32;
    const uint lane = local_id % 32;

    const uint k_vec4 = p.K >> 2;
    for (uint i = local_id; i < k_vec4; i += TG_SIZE) {
        x_cache4[i] = *(device const float4*)(input + (i << 2));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tg_pos.x * ROWS_PER_TG + sg_idx;
    if (row >= p.M) return;

    const uint bpr = p.K / 256;
    const ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);
    device const uchar* row_ptr = W + expert_base + ulong(row) * ulong(bpr) * 144;

    float acc = 0.0f;

    for (uint bi = 0; bi < bpr; bi++) {
        device const uchar* block = row_ptr + bi * 144;

        const float d = float(as_type<half>(*(device const ushort*)(block)));
        const float dmin = float(as_type<half>(*(device const ushort*)(block + 2)));

        device const uchar* scales = block + 4;
        device const uchar* quants = block + 16;

        const uint byte_off = lane * 4;
        const uint j = byte_off / 32;
        const uint local_off = byte_off % 32;

        const uchar4 qbytes = *(device const uchar4*)(quants + byte_off);

        const float2 sm_lo = get_scale_min_k4(j * 2, scales);
        const float2 sm_hi = get_scale_min_k4(j * 2 + 1, scales);

        const float d_sc_lo = d * sm_lo.x;
        const float d_m_lo = dmin * sm_lo.y;
        const float d_sc_hi = d * sm_hi.x;
        const float d_m_hi = dmin * sm_hi.y;

        const uint col_lo = bi * 256 + j * 64 + local_off;
        const uint col_hi = col_lo + 32;

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

        acc += dot(lo_vals, x_lo);
        acc += dot(hi_vals, x_hi);
    }

    const float sum = simd_sum(acc);
    if (lane == 0) {
        Y[p.y_offset / 4 + expert_slot * p.M + row] = sum;
    }
}
