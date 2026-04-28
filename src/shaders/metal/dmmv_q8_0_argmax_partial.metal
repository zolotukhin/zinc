#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Q8_0 LM-head greedy argmax, stage 1.
//
// Adapted from llama.cpp ggml-metal.metal::kernel_mul_mv_q8_0_f32's
// two-rows-per-simdgroup Q8_0 row mapping. Instead of writing all logits,
// each simdgroup writes its covered rows as (token_id, logit) pairs. A second
// kernel reduces those row-level pairs to the final greedy token id.

kernel void main0(
    constant DmmvPush& p             [[buffer(0)]],
    device const uchar* W            [[buffer(1)]],
    device const float* X            [[buffer(2)]],
    device uint2* partials           [[buffer(3)]],
    uint tg_id                       [[threadgroup_position_in_grid]],
    uint sg_idx                      [[simdgroup_index_in_threadgroup]],
    uint lane                        [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg           [[simdgroups_per_threadgroup]]
) {
    const uint base_row = (tg_id * simdgroups_per_tg + sg_idx) * 2u;
    const bool valid0 = base_row < p.M;
    const bool valid1 = (base_row + 1u) < p.M;

    device const float* input = X + (p.x_offset >> 2);
    const uint blocks_per_row = p.K >> 5;
    const ulong row_bytes = ulong(blocks_per_row) * 34ull;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    if (valid0 || valid1) {
        device const uchar* row0 = W + p.a_offset + ulong(base_row) * row_bytes;
        device const uchar* row1 = row0 + row_bytes;

        for (uint bi = lane; bi < blocks_per_row; bi += 32u) {
            const uint x_base = bi << 5;

            if (valid0) {
                device const uchar* blk0 = row0 + bi * 34u;
                const float s0 = float(as_type<half>(*(device const ushort*)(blk0)));
                device const packed_char4* q0 = (device const packed_char4*)(blk0 + 2u);

                #pragma unroll
                for (uint vi = 0u; vi < 8u; ++vi) {
                    const float4 x = *(device const float4*)(input + x_base + (vi << 2));
                    acc0 = fma(s0, dot(float4(char4(q0[vi])), x), acc0);
                }
            }

            if (valid1) {
                device const uchar* blk1 = row1 + bi * 34u;
                const float s1 = float(as_type<half>(*(device const ushort*)(blk1)));
                device const packed_char4* q1 = (device const packed_char4*)(blk1 + 2u);

                #pragma unroll
                for (uint vi = 0u; vi < 8u; ++vi) {
                    const float4 x = *(device const float4*)(input + x_base + (vi << 2));
                    acc1 = fma(s1, dot(float4(char4(q1[vi])), x), acc1);
                }
            }
        }
    }

    const float sum0 = simd_sum(acc0);
    const float sum1 = simd_sum(acc1);

    if (lane == 0u) {
        if (valid0) {
            partials[base_row] = uint2(base_row, as_type<uint>(sum0));
        }
        if (valid1) {
            partials[base_row + 1u] = uint2(base_row + 1u, as_type<uint>(sum1));
        }
    }
}
