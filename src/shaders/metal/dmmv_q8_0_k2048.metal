#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// K <= 2048 specialization of Q8_0 DMMV.
//
// The generic dmmv_q8_0 kernel declares half4[1024] = 8 KiB of threadgroup
// memory to handle K up to 4096.  For the dominant decode-side Q8_0 operations
// (SSM projections M=8192/4096 K=2048, LM-head M=248320 K=2048, router, shared
// experts), only half4[512] = 4 KiB is needed.
//
// Halving the threadgroup memory footprint doubles the number of threadgroups
// that can reside on each GPU core simultaneously (from 4 to 8 on Apple9/M4
// with 32 KiB tile memory), improving latency hiding and effective memory
// bandwidth utilization.
//
// This mirrors the existing dmmv_q4k_k2048 pattern.

kernel void main0(
    constant DmmvPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    // 512 half4 = 4 KiB — half the generic kernel's footprint.
    threadgroup half4 x_cache4[512];

    const uint tg_size = simdgroups_per_tg * 32u;
    device const float* input = X + (p.x_offset >> 2);
    device float* output = Y + (p.y_offset >> 2);

    const uint k_vec4 = p.K >> 2;
    for (uint i = local_id; i < k_vec4; i += tg_size) {
        x_cache4[i] = half4(*(device const float4*)(input + (i << 2)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tg_id * simdgroups_per_tg + sg_idx;
    if (row >= p.M) return;

    const uint blocks_per_row = p.K >> 5;
    device const uchar* row_ptr = W + p.a_offset + ulong(row) * ulong(blocks_per_row) * 34ull;

    float acc = 0.0f;
    for (uint bi = lane; bi < blocks_per_row; bi += 32u) {
        device const uchar* block = row_ptr + bi * 34u;
        const float scale = float(as_type<half>(*(device const ushort*)(block)));
        device const packed_char4* quants = (device const packed_char4*)(block + 2u);
        const uint x_base = bi * 8u;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const char4 q = char4(quants[vi]);
            const half4 x = x_cache4[x_base + vi];
            const half4 q_half = half4(q);
            acc = fma(scale, float(dot(q_half, x)), acc);
        }
    }

    const float sum = simd_sum(acc);
    if (lane == 0u) {
        output[row] = sum;
    }
}
