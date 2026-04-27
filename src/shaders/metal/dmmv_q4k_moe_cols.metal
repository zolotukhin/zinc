#include <metal_stdlib>
using namespace metal;

// Q4_K grouped MoE DMMV for batched Gemma fused gate/up prefill.
//
// Dispatch: grid (ceil(M / 2), n_experts, ceil(max_count / 4)),
// threadgroup (64, 1, 1). grid.y is the real expert id. grid.z selects a
// packed route-id block from moe_route_pack.metal's [expert][ids_stride] table.
//
// Each simdgroup owns one output row and computes up to four routed token
// vectors for that row while reading/dequantizing the Q4_K weights once.

struct MoeColsDmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint expert_stride;
    uint x_offset;
    uint y_offset;
    uint ids_stride;
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

#define NUM_COLS 4u

kernel void main0(
    device const uchar* W                     [[buffer(0)]],
    constant MoeColsDmmvPush& p               [[buffer(1)]],
    device const float* X                     [[buffer(2)]],
    device float* Y                           [[buffer(3)]],
    device const uint* counts                 [[buffer(4)]],
    device const uint* packed_ids             [[buffer(5)]],
    uint3 tg_pos                              [[threadgroup_position_in_grid]],
    uint tid                                  [[thread_index_in_simdgroup]],
    uint sgid                                 [[simdgroup_index_in_threadgroup]]
) {
    const uint expert_id = tg_pos.y;
    const uint row = tg_pos.x * 2u + sgid;
    if (row >= p.M) {
        return;
    }

    const uint packed_base = tg_pos.z * NUM_COLS;
    const uint count = counts[expert_id];
    if (packed_base >= count) {
        return;
    }

    const bool active0 = packed_base + 0u < count;
    const bool active1 = packed_base + 1u < count;
    const bool active2 = packed_base + 2u < count;
    const bool active3 = packed_base + 3u < count;

    device const uint* expert_ids = packed_ids + expert_id * p.ids_stride;
    const uint route0 = active0 ? expert_ids[packed_base + 0u] : 0u;
    const uint route1 = active1 ? expert_ids[packed_base + 1u] : 0u;
    const uint route2 = active2 ? expert_ids[packed_base + 2u] : 0u;
    const uint route3 = active3 ? expert_ids[packed_base + 3u] : 0u;

    device const float* x_base = X + (p.x_offset / 4u);
    device const float* x0 = x_base + route0 * p.K;
    device const float* x1 = x_base + route1 * p.K;
    device const float* x2 = x_base + route2 * p.K;
    device const float* x3 = x_base + route3 * p.K;

    const uint nb = p.K / 256u;
    const uint bpb = 144u;
    const ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);
    device const uchar* src = W + expert_base + ulong(row) * ulong(nb) * ulong(bpb);

    float4 acc = float4(0.0f);

    for (uint b = 0u; b < nb; b++) {
        device const uchar* block = src + b * bpb;

        const float d = float(as_type<half>(*(device const ushort*)(block)));
        const float dmin = float(as_type<half>(*(device const ushort*)(block + 2)));
        device const uchar* scales = block + 4;
        device const uchar* quants = block + 16;

        const uint byte_off = tid * 4u;
        const uint j = byte_off / 32u;
        const uint local_off = byte_off % 32u;

        const uchar4 qbytes = *(device const uchar4*)(quants + byte_off);
        const float2 sm_lo = get_scale_min_k4(j * 2u, scales);
        const float2 sm_hi = get_scale_min_k4(j * 2u + 1u, scales);

        const float d_sc_lo = d * sm_lo.x;
        const float d_m_lo = dmin * sm_lo.y;
        const float d_sc_hi = d * sm_hi.x;
        const float d_m_hi = dmin * sm_hi.y;

        const uint col_lo = b * 256u + j * 64u + local_off;
        const uint col_hi = col_lo + 32u;

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

        if (active0) {
            const float4 x_lo = *(device const float4*)(x0 + col_lo);
            const float4 x_hi = *(device const float4*)(x0 + col_hi);
            acc.x += dot(lo_vals, x_lo) + dot(hi_vals, x_hi);
        }
        if (active1) {
            const float4 x_lo = *(device const float4*)(x1 + col_lo);
            const float4 x_hi = *(device const float4*)(x1 + col_hi);
            acc.y += dot(lo_vals, x_lo) + dot(hi_vals, x_hi);
        }
        if (active2) {
            const float4 x_lo = *(device const float4*)(x2 + col_lo);
            const float4 x_hi = *(device const float4*)(x2 + col_hi);
            acc.z += dot(lo_vals, x_lo) + dot(hi_vals, x_hi);
        }
        if (active3) {
            const float4 x_lo = *(device const float4*)(x3 + col_lo);
            const float4 x_hi = *(device const float4*)(x3 + col_hi);
            acc.w += dot(lo_vals, x_lo) + dot(hi_vals, x_hi);
        }
    }

    const float out0 = simd_sum(acc.x);
    const float out1 = simd_sum(acc.y);
    const float out2 = simd_sum(acc.z);
    const float out3 = simd_sum(acc.w);

    device float* y_base = Y + (p.y_offset / 4u);
    if (tid == 0u) {
        if (active0) y_base[route0 * p.M + row] = out0;
        if (active1) y_base[route1 * p.M + row] = out1;
        if (active2) y_base[route2 * p.M + row] = out2;
        if (active3) y_base[route3 * p.M + row] = out3;
    }
}
