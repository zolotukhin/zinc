#include <metal_stdlib>
using namespace metal;

// Q5_1 grouped MoE DMMV for batched Gemma expert-down prefill.
//
// Dispatch: grid (ceil(M / 2), n_experts, ceil(max_count / 4)),
// threadgroup (64, 1, 1). grid.y is the real expert id. grid.z selects a
// packed route-id block from moe_route_pack.metal's [expert][ids_stride] table.
//
// Each simdgroup owns one output row and computes up to four routed token
// vectors for that row while reading/dequantizing the Q5_1 weights once.

struct MoeColsDmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint expert_stride;
    uint x_offset;
    uint y_offset;
    uint ids_stride;
};

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

    const uint nb = p.K / 32u;
    const uint bpb = 24u;
    const ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);
    device const uchar* src = W + expert_base + ulong(row) * ulong(nb) * ulong(bpb);

    float4 acc = float4(0.0f);

    for (uint b = tid; b < nb; b += 32u) {
        device const uchar* block = src + b * bpb;

        const float d = float(*((device const half*)block));
        const float m = float(*((device const half*)(block + 2)));
        const uint qh = uint(block[4]) | (uint(block[5]) << 8)
                      | (uint(block[6]) << 16) | (uint(block[7]) << 24);
        device const uchar* qs = block + 8;
        const uint base = b * 32u;

        float4 sum_qx = float4(0.0f);
        float4 sum_x = float4(0.0f);
        for (uint j = 0u; j < 16u; j++) {
            const uchar q_byte = qs[j];
            const uint lo = q_byte & 0x0F;
            const uint hi = q_byte >> 4;
            const uint q0 = lo | (((qh >> j) & 1u) << 4);
            const uint q1 = hi | (((qh >> (j + 16u)) & 1u) << 4);

            const float4 x_lo = float4(
                active0 ? x0[base + j] : 0.0f,
                active1 ? x1[base + j] : 0.0f,
                active2 ? x2[base + j] : 0.0f,
                active3 ? x3[base + j] : 0.0f
            );
            const float4 x_hi = float4(
                active0 ? x0[base + 16u + j] : 0.0f,
                active1 ? x1[base + 16u + j] : 0.0f,
                active2 ? x2[base + 16u + j] : 0.0f,
                active3 ? x3[base + 16u + j] : 0.0f
            );

            sum_qx += float(q0) * x_lo + float(q1) * x_hi;
            sum_x += x_lo + x_hi;
        }

        acc += d * sum_qx + m * sum_x;
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
