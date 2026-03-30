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

inline float fp16_to_fp32(uint h) {
    return float(as_type<half>(ushort(h)));
}

kernel void main0(
    device const uchar* W [[buffer(0)]],
    constant MoeDmmvPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    device const uint* expert_ids [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tg_pos [[threadgroup_position_in_grid]]
) {
    const uint row = gid.x;
    if (row >= p.M) return;

    const uint expert_slot = tg_pos.y;
    const uint expert_id = expert_ids[expert_slot];
    const uint blocks_per_row = p.K / 256u;
    const uint row_offset = p.a_offset + expert_id * p.expert_stride + row * blocks_per_row * 176u;
    const uint x_base = (p.x_offset / 4u) + expert_slot * p.x_expert_stride;
    const uint y_base = (p.y_offset / 4u) + expert_slot * p.M;

    float sum = 0.0f;
    uint xi = 0u;

    for (uint b = 0u; b < blocks_per_row; b++) {
        const uint bb = row_offset + b * 176u;

        const uint d_bits = uint(W[bb]) | (uint(W[bb + 1u]) << 8u);
        const float d = fp16_to_fp32(d_bits);
        const uint dm_bits = uint(W[bb + 2u]) | (uint(W[bb + 3u]) << 8u);
        const float dmin = fp16_to_fp32(dm_bits);

        for (uint g = 0u; g < 4u; g++) {
            const uint sb_lo = g * 2u;
            const uint sb_hi = sb_lo + 1u;

            uint sc_lo = 0u;
            uint m_lo = 0u;
            uint sc_hi = 0u;
            uint m_hi = 0u;

            if (sb_lo < 4u) {
                sc_lo = uint(W[bb + 4u + sb_lo]) & 63u;
                m_lo = uint(W[bb + 8u + sb_lo]) & 63u;
            } else {
                sc_lo = (uint(W[bb + 8u + sb_lo]) & 0x0Fu) | ((uint(W[bb + sb_lo]) >> 6u) << 4u);
                m_lo = (uint(W[bb + 8u + sb_lo]) >> 4u) | ((uint(W[bb + 4u + sb_lo]) >> 6u) << 4u);
            }

            if (sb_hi < 4u) {
                sc_hi = uint(W[bb + 4u + sb_hi]) & 63u;
                m_hi = uint(W[bb + 8u + sb_hi]) & 63u;
            } else {
                sc_hi = (uint(W[bb + 8u + sb_hi]) & 0x0Fu) | ((uint(W[bb + sb_hi]) >> 6u) << 4u);
                m_hi = (uint(W[bb + 8u + sb_hi]) >> 4u) | ((uint(W[bb + 4u + sb_hi]) >> 6u) << 4u);
            }

            const float factor_lo = d * float(sc_lo);
            const float bias_lo = dmin * float(m_lo);
            const float factor_hi = d * float(sc_hi);
            const float bias_hi = dmin * float(m_hi);

            const uint qs_base = bb + 48u + g * 32u;
            const uint x_grp = x_base + xi;

            for (uint e = 0u; e < 32u; e++) {
                const uint byte_val = uint(W[qs_base + e]);
                const uint q_lo = byte_val & 0x0Fu;
                const uint q_hi = byte_val >> 4u;
                const uint qh_val = uint(W[bb + 16u + e]);
                const uint hb_lo = (qh_val >> sb_lo) & 1u;
                const uint hb_hi = (qh_val >> sb_hi) & 1u;

                const float v_lo = float(q_lo | (hb_lo << 4u));
                const float v_hi = float(q_hi | (hb_hi << 4u));

                sum += (factor_lo * v_lo - bias_lo) * X[x_grp + e];
                sum += (factor_hi * v_hi - bias_hi) * X[x_grp + 32u + e];
            }

            xi += 64u;
        }
    }

    Y[y_base + row] = sum;
}
