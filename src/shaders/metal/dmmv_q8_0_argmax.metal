#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Q8_0 LM-head greedy path: keep llama.cpp's two-rows-per-simdgroup
// mul_mv shape, but emit one (idx, value) pair per threadgroup instead of
// materializing the full vocab logits.

kernel void main0(
    constant DmmvPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device uint* partials [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    threadgroup float best_vals[32];
    threadgroup uint best_idxs[32];

    const uint base_row = (tg_id * simdgroups_per_tg + sg_idx) * 2u;
    const bool active0 = base_row < p.M;
    const bool active1 = base_row + 1u < p.M;

    device const float* input = X + (p.x_offset >> 2);

    const uint blocks_per_row = p.K >> 5;
    const ulong row_bytes = ulong(blocks_per_row) * 34ull;
    device const uchar* row0 = W + p.a_offset + ulong(base_row) * row_bytes;
    device const uchar* row1 = row0 + row_bytes;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint bi = lane; bi < blocks_per_row; bi += 32u) {
        const uint x_base = bi << 5;

        if (active0) {
            device const uchar* blk0 = row0 + bi * 34u;
            const float s0 = float(as_type<half>(*(device const ushort*)(blk0)));
            device const packed_char4* q0 = (device const packed_char4*)(blk0 + 2u);
            #pragma unroll
            for (uint vi = 0u; vi < 8u; ++vi) {
                const float4 x = *(device const float4*)(input + x_base + (vi << 2));
                acc0 = fma(s0, dot(float4(char4(q0[vi])), x), acc0);
            }
        }

        if (active1) {
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

    float best_val = active0 ? simd_sum(acc0) : -INFINITY;
    uint best_idx = active0 ? base_row : 0u;
    if (active1) {
        const float sum1 = simd_sum(acc1);
        if (sum1 > best_val) {
            best_val = sum1;
            best_idx = base_row + 1u;
        }
    }

    if (lane == 0u) {
        best_vals[sg_idx] = best_val;
        best_idxs[sg_idx] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float tg_best_val = best_vals[0];
        uint tg_best_idx = best_idxs[0];
        for (uint i = 1u; i < simdgroups_per_tg; ++i) {
            const float v = best_vals[i];
            const uint idx = best_idxs[i];
            if (v > tg_best_val || (v == tg_best_val && idx < tg_best_idx)) {
                tg_best_val = v;
                tg_best_idx = idx;
            }
        }
        partials[tg_id * 2u + 0u] = tg_best_idx;
        partials[tg_id * 2u + 1u] = as_type<uint>(tg_best_val);
    }
}
