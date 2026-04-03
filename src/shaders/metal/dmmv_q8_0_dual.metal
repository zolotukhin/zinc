#include <metal_stdlib>
using namespace metal;

struct DualQ8DmmvPush {
    uint M0;
    uint M1;
    uint K;
    uint a0_offset;
    uint a1_offset;
    uint x_offset;
    uint y0_offset;
    uint y1_offset;
};

// Dual-output q8_0 decode path for SSM pre-projections that share the same
// normalized hidden-state input. This stages X once per workgroup and then
// routes each row to one of two q8 matrices, reducing repeated activation
// fetches on the 8192x2048 + 4096x2048 pair that dominates SSM token time.
kernel void main0(
    constant DualQ8DmmvPush& p [[buffer(0)]],
    device const uchar* W0 [[buffer(1)]],
    device const uchar* W1 [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* Y0 [[buffer(4)]],
    device float* Y1 [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    threadgroup half4 x_cache4[512];

    const uint tg_size = simdgroups_per_tg * 32u;
    device const float* input = X + (p.x_offset >> 2);
    const uint k_vec4 = p.K >> 2;
    for (uint i = local_id; i < k_vec4; i += tg_size) {
        x_cache4[i] = half4(*(device const float4*)(input + (i << 2)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint linear_row = tg_id * simdgroups_per_tg + sg_idx;
    const uint total_rows = p.M0 + p.M1;
    if (linear_row >= total_rows) return;

    const bool first = linear_row < p.M0;
    const uint row = first ? linear_row : (linear_row - p.M0);
    device const uchar* weights = first ? W0 : W1;
    device float* output = first ? (Y0 + (p.y0_offset >> 2)) : (Y1 + (p.y1_offset >> 2));
    const uint a_offset = first ? p.a0_offset : p.a1_offset;

    const uint blocks_per_row = p.K >> 5;
    device const uchar* row_ptr = weights + a_offset + ulong(row) * ulong(blocks_per_row) * 34ull;

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
