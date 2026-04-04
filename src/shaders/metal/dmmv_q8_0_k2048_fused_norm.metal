#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Fused RMSNorm + Q8_0 DMMV for K <= 2048 — eliminates separate norm dispatch + barrier.
//
// Instead of reading a pre-normalized input vector from norm_buf, each simdgroup
// independently computes the RMSNorm factor from the raw hidden state.  Hidden
// (8 KiB for K=2048) and norm_weight are L1-cached after the first simdgroup on
// a core accesses them; subsequent simdgroups get L1 hits.  This removes one
// barrier and one dispatch per SSM layer (30 per decode step on Qwen3.5-35B-A3B).
//
// The inline norm adds ~64 FMAs + simd_sum + rsqrt per simdgroup — negligible
// compared to the weight memory reads that dominate DMMV execution time.

kernel void main0(
    constant DmmvPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* hidden [[buffer(2)]],
    device float* Y [[buffer(3)]],
    device const float* norm_weight [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    device const float* h = hidden + (p.x_offset >> 2);
    device float* output = Y + (p.y_offset >> 2);

    // Step 1: Compute RMS normalization factor from raw hidden state.
    // All simdgroups compute this independently — hidden is L1-cached.
    float sq_sum = 0.0f;
    for (uint i = lane; i < p.K; i += 32u) {
        const float v = h[i];
        sq_sum += v * v;
    }
    sq_sum = simd_sum(sq_sum);
    const float rms_inv = rsqrt(sq_sum / float(p.K) + 1e-6f);

    const uint row = tg_id * simdgroups_per_tg + sg_idx;
    if (row >= p.M) return;

    // Step 2: DMMV with inline-normalized input.
    const uint blocks_per_row = p.K >> 5;
    device const uchar* row_ptr = W + p.a_offset + ulong(row) * ulong(blocks_per_row) * 34ull;

    float acc = 0.0f;
    for (uint bi = lane; bi < blocks_per_row; bi += 32u) {
        device const uchar* block = row_ptr + bi * 34u;
        const float scale = float(as_type<half>(*(device const ushort*)(block)));
        device const packed_char4* quants = (device const packed_char4*)(block + 2u);
        const uint x_base = bi << 5;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const char4 q = char4(quants[vi]);
            const half4 q_half = half4(q);
            const uint idx = x_base + (vi << 2);
            // Inline RMSNorm: x[i] = norm_weight[i] * (hidden[i] * rms_inv)
            const float4 h4 = *(device const float4*)(h + idx);
            const float4 nw4 = *(device const float4*)(norm_weight + idx);
            const half4 x = half4(nw4 * (h4 * rms_inv));
            acc = fma(scale, float(dot(q_half, x)), acc);
        }
    }

    const float sum = simd_sum(acc);
    if (lane == 0u) {
        output[row] = sum;
    }
}
