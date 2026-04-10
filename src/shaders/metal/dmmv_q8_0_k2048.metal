#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// K <= 2048 specialization of Q8_0 DMMV — nr=2 multi-row, barrier-free.
//
// Adapted from llama.cpp kernel_mul_mv_q8_0_f32 (ggml-metal.metal) which uses
// N_R0_Q8_0 = 2: each simdgroup processes TWO output rows simultaneously,
// sharing the L1-cached X vector (4-8 KiB for K<=2048).  This doubles useful
// compute per X fetch, improving pipeline utilization and bandwidth efficiency.
//
// Weight data is loaded via aligned int32_t reads (4 bytes at once) instead of
// individual byte loads, matching llama.cpp's char4(*(int*)) pattern.

kernel void main0(
    constant DmmvPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    const uint base_row = (tg_id * simdgroups_per_tg + sg_idx) * 2u;
    if (base_row >= p.M) return;

    device const float* input = X + (p.x_offset >> 2);
    device float* output = Y + (p.y_offset >> 2);

    const uint blocks_per_row = p.K >> 5;
    const ulong row_bytes = ulong(blocks_per_row) * 34ull;
    device const uchar* row0 = W + p.a_offset + ulong(base_row) * row_bytes;
    device const uchar* row1 = row0 + row_bytes;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint bi = lane; bi < blocks_per_row; bi += 32u) {
        device const uchar* blk0 = row0 + bi * 34u;
        device const uchar* blk1 = row1 + bi * 34u;
        const float s0 = float(as_type<half>(*(device const ushort*)(blk0)));
        const float s1 = float(as_type<half>(*(device const ushort*)(blk1)));
        const uint x_base = bi << 5;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const uint qo = 2u + (vi << 2);
            const float4 x = *(device const float4*)(input + x_base + (vi << 2));
            acc0 = fma(s0, dot(float4(as_type<char4>(*(device const int*)(blk0 + qo))), x), acc0);
            acc1 = fma(s1, dot(float4(as_type<char4>(*(device const int*)(blk1 + qo))), x), acc1);
        }
    }

    const float sum0 = simd_sum(acc0);
    if (lane == 0u) output[base_row] = sum0;

    if (base_row + 1u < p.M) {
        const float sum1 = simd_sum(acc1);
        if (lane == 0u) output[base_row + 1u] = sum1;
    }
}
