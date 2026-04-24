#include <metal_stdlib>
using namespace metal;

// Q5_1 batched MoE DMMV: replays dmmv_q5_1.metal per expert_slot, reading the
// per-slot expert ID from the routing buffer (router_output_buf written by
// softmax_topk or by the Gemma fallback's CPU topKSoftmax).
//
// Dispatch: grid (rows / 2, n_experts_used, 1), threadgroup (64, 1, 1).
// Each workgroup handles 2 rows of one expert (matches the per-token shader's
// 2-rows-per-WG layout). All experts share the same input vector slice in X
// (or n_experts_used × inter_dim slices when x_expert_stride != 0).

struct MoeDmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint expert_stride;
    uint x_expert_stride;
    uint x_offset;
    uint y_offset;
};

kernel void main0(
    device const uchar* W                     [[buffer(0)]],
    constant MoeDmmvPush& p                   [[buffer(1)]],
    device const float* X                     [[buffer(2)]],
    device float* Y                           [[buffer(3)]],
    device const uint* expert_ids             [[buffer(4)]],
    uint3 tg_pos                              [[threadgroup_position_in_grid]],
    uint tid                                  [[thread_index_in_simdgroup]],
    uint sgid                                 [[simdgroup_index_in_threadgroup]]
) {
    const uint expert_slot = tg_pos.y;
    const uint expert_id   = expert_ids[expert_slot];

    // 32 threads per simdgroup, 2 simdgroups per threadgroup => 2 rows per WG.
    const uint row = tg_pos.x * 2 + sgid;
    if (row >= p.M) return;

    const uint nb  = p.K / 32;     // Q5_1 blocks per row
    const uint bpb = 24;           // bytes per Q5_1 block

    // Per-expert weight base offset.
    ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);
    device const uchar* src = W + expert_base + ulong(row) * ulong(nb) * ulong(bpb);

    // Per-expert input slice (x_expert_stride is in float elements).
    device const float* x = X + (p.x_offset / 4) + expert_slot * p.x_expert_stride;

    float sum = 0.0f;

    for (uint b = tid; b < nb; b += 32) {
        device const uchar* block = src + b * bpb;

        const float d = float(*((device const half*)block));
        const float m = float(*((device const half*)(block + 2)));

        const uint qh = uint(block[4]) | (uint(block[5]) << 8)
                      | (uint(block[6]) << 16) | (uint(block[7]) << 24);

        device const uchar* qs = block + 8;
        const uint base = b * 32;

        float sum_qx = 0.0f;
        float sum_x  = 0.0f;
        for (uint j = 0; j < 16; j++) {
            const uchar q_byte = qs[j];
            const uint lo = q_byte & 0x0F;
            const uint hi = q_byte >> 4;

            const uint bit_lo = (qh >> j)        & 1;
            const uint bit_hi = (qh >> (j + 16)) & 1;

            const uint q0 = lo | (bit_lo << 4);
            const uint q1 = hi | (bit_hi << 4);

            const float x0 = x[base + j];
            const float x1 = x[base + 16 + j];

            sum_qx += float(q0) * x0 + float(q1) * x1;
            sum_x  += x0 + x1;
        }

        sum += d * sum_qx + m * sum_x;
    }

    sum = simd_sum(sum);

    if (tid == 0) {
        Y[(p.y_offset / 4) + expert_slot * p.M + row] = sum;
    }
}
