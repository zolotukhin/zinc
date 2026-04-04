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

inline float2 get_scale_min_k5(uint j, device const uchar* scales) {
    if (j < 4u) {
        return float2(float(scales[j] & 63u), float(scales[4u + j] & 63u));
    }
    return float2(
        float((scales[4u + j] & 0x0Fu) | ((scales[j - 4u] >> 6u) << 4u)),
        float((scales[4u + j] >> 4u) | ((scales[j] >> 6u) << 4u))
    );
}

// K <= 2048 specialization of batched MoE Q5_K DMMV — barrier-free, L1-cached X reads.
//
// Previous version staged X into threadgroup memory and synchronized all
// simdgroups with threadgroup_barrier.  This couples simdgroups within a
// threadgroup: none can start computing until ALL finish loading X.
//
// This version eliminates threadgroup memory entirely.  Each simdgroup reads
// X directly from device memory; after the first access on a core the <= 8 KiB
// vector lands in L1 cache and subsequent simdgroups get L1 hits.  All
// simdgroups become fully independent, increasing concurrent memory streams
// and improving bandwidth utilization.
#define TG_SIZE 512
#define ROWS_PER_TG (TG_SIZE / 32)

kernel void main0(
    device const uchar* W [[buffer(0)]],
    constant MoeDmmvPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    device const uint* expert_ids [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint expert_slot = tg_pos.y;
    const uint expert_id = expert_ids[expert_slot];
    device const float* input = X + (p.x_offset / 4u) + expert_slot * p.x_expert_stride;

    const uint row = tg_pos.x * ROWS_PER_TG + sg_idx;
    if (row >= p.M) return;

    const uint blocks_per_row = p.K / 256u;
    ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);
    device const uchar* row_ptr = W + expert_base + ulong(row) * ulong(blocks_per_row) * 176ull;

    float sum = 0.0f;

    for (uint b = 0u; b < blocks_per_row; b++) {
        device const uchar* block = row_ptr + b * 176u;

        const float d = float(as_type<half>(*(device const ushort*)(block)));
        const float dmin = float(as_type<half>(*(device const ushort*)(block + 2)));
        device const uchar* scales = block + 4u;
        device const uchar* high_bits = block + 16u;
        device const uchar* quants = block + 48u;

        const uint qh_val = uint(high_bits[lane]);
        const uint col_base = b * 256u;

        #pragma unroll
        for (uint g = 0u; g < 4u; g++) {
            const uint sb_lo = g * 2u;
            const uint sb_hi = g * 2u + 1u;
            const float2 sm_lo = get_scale_min_k5(sb_lo, scales);
            const float2 sm_hi = get_scale_min_k5(sb_hi, scales);
            const float factor_lo = d * sm_lo.x;
            const float bias_lo = dmin * sm_lo.y;
            const float factor_hi = d * sm_hi.x;
            const float bias_hi = dmin * sm_hi.y;

            const uint q_byte = uint(quants[g * 32u + lane]);
            const float v_lo = factor_lo * float((q_byte & 0x0Fu) | (((qh_val >> sb_lo) & 1u) << 4u)) - bias_lo;
            const float v_hi = factor_hi * float((q_byte >> 4u) | (((qh_val >> sb_hi) & 1u) << 4u)) - bias_hi;

            const uint col_lo = col_base + g * 64u + lane;
            const uint col_hi = col_lo + 32u;

            sum += v_lo * input[col_lo];
            sum += v_hi * input[col_hi];
        }
    }

    const float total = simd_sum(sum);
    if (lane == 0u) {
        Y[p.y_offset / 4u + expert_slot * p.M + row] = total;
    }
}
