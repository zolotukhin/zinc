#include <metal_stdlib>
using namespace metal;

// Q5_1 batched MoE DMMV: replays dmmv_q5_1.metal per expert_slot, reading the
// per-slot expert ID from the routing buffer (router_output_buf written by
// softmax_topk or by the Gemma fallback's CPU topKSoftmax).
//
// Dispatch: grid (ceil(rows / 4), n_experts_used, 1), threadgroup (64, 1, 1).
// Each workgroup handles 4 rows of one expert: each simdgroup computes two
// adjacent rows while sharing the same cached activation vector. All experts
// share the same input vector slice in X
// (or n_experts_used × inter_dim slices when x_expert_stride != 0).
//
// Gemma's expert-down shape has K=704, so the two rows in a workgroup used to
// reread the same activation vector from device memory. Cache the vector once
// per workgroup while keeping the existing 64-thread layout.

struct MoeDmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint expert_stride;
    uint x_expert_stride;
    uint x_offset;
    uint y_offset;
};

#define X_CACHE_MAX 4096

kernel void main0(
    device const uchar* W                     [[buffer(0)]],
    constant MoeDmmvPush& p                   [[buffer(1)]],
    device const float* X                     [[buffer(2)]],
    device float* Y                           [[buffer(3)]],
    device const uint* expert_ids             [[buffer(4)]],
    uint3 tg_pos                              [[threadgroup_position_in_grid]],
    uint3 local_pos                           [[thread_position_in_threadgroup]],
    uint tid                                  [[thread_index_in_simdgroup]],
    uint sgid                                 [[simdgroup_index_in_threadgroup]]
) {
    const uint expert_slot = tg_pos.y;
    const uint expert_id   = expert_ids[expert_slot];

    // 32 threads per simdgroup, 2 simdgroups per threadgroup, 2 rows per
    // simdgroup => 4 rows per WG. Keep all threads alive through the cache
    // barrier even on tail workgroups.
    const uint row0 = tg_pos.x * 4u + sgid * 2u;
    const uint row1 = row0 + 1u;
    const bool valid0 = row0 < p.M;
    const bool valid1 = row1 < p.M;

    const uint nb  = p.K / 32;     // Q5_1 blocks per row
    const uint bpb = 24;           // bytes per Q5_1 block

    // Per-expert weight base offset.
    const ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);
    const ulong row_bytes = ulong(nb) * ulong(bpb);
    device const uchar* src0 = W + expert_base + ulong(row0) * row_bytes;
    device const uchar* src1 = W + expert_base + ulong(row1) * row_bytes;

    // Per-expert input slice (x_expert_stride is in float elements).
    device const float* x = X + (p.x_offset / 4) + expert_slot * p.x_expert_stride;
    threadgroup float x_cache[X_CACHE_MAX];
    const bool use_cache = p.K <= X_CACHE_MAX;
    if (use_cache) {
        for (uint i = local_pos.x; i < p.K; i += 64u) {
            x_cache[i] = x[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!valid0 && !valid1) {
        return;
    }

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (uint b = tid; b < nb; b += 32) {
        device const uchar* block0 = src0 + b * bpb;
        device const uchar* block1 = src1 + b * bpb;

        const float d0 = float(*((device const half*)block0));
        const float m0 = float(*((device const half*)(block0 + 2)));
        const uint qh0 = uint(block0[4]) | (uint(block0[5]) << 8)
                       | (uint(block0[6]) << 16) | (uint(block0[7]) << 24);
        device const uchar* qs0 = block0 + 8;

        float d1 = 0.0f;
        float m1 = 0.0f;
        uint qh1 = 0u;
        device const uchar* qs1 = qs0;
        if (valid1) {
            d1 = float(*((device const half*)block1));
            m1 = float(*((device const half*)(block1 + 2)));
            qh1 = uint(block1[4]) | (uint(block1[5]) << 8)
                | (uint(block1[6]) << 16) | (uint(block1[7]) << 24);
            qs1 = block1 + 8;
        }

        const uint base = b * 32;

        float sum_qx0 = 0.0f;
        float sum_qx1 = 0.0f;
        float sum_x  = 0.0f;
        for (uint j = 0; j < 16; j++) {
            const uchar q_byte0 = qs0[j];
            const uint lo0 = q_byte0 & 0x0F;
            const uint hi0 = q_byte0 >> 4;

            const uint bit_lo0 = (qh0 >> j)        & 1;
            const uint bit_hi0 = (qh0 >> (j + 16)) & 1;

            const uint q00 = lo0 | (bit_lo0 << 4);
            const uint q01 = hi0 | (bit_hi0 << 4);

            const float x0 = use_cache ? x_cache[base + j] : x[base + j];
            const float x1 = use_cache ? x_cache[base + 16 + j] : x[base + 16 + j];

            sum_qx0 += float(q00) * x0 + float(q01) * x1;
            sum_x  += x0 + x1;

            if (valid1) {
                const uchar q_byte1 = qs1[j];
                const uint lo1 = q_byte1 & 0x0F;
                const uint hi1 = q_byte1 >> 4;
                const uint bit_lo1 = (qh1 >> j)        & 1;
                const uint bit_hi1 = (qh1 >> (j + 16)) & 1;
                const uint q10 = lo1 | (bit_lo1 << 4);
                const uint q11 = hi1 | (bit_hi1 << 4);
                sum_qx1 += float(q10) * x0 + float(q11) * x1;
            }
        }

        sum0 += d0 * sum_qx0 + m0 * sum_x;
        if (valid1) {
            sum1 += d1 * sum_qx1 + m1 * sum_x;
        }
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);

    if (tid == 0) {
        device float* out = Y + (p.y_offset / 4) + expert_slot * p.M;
        if (valid0) out[row0] = sum0;
        if (valid1) out[row1] = sum1;
    }
}
