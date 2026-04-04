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

// Fused RMSNorm + Dual-output Q8_0 DMMV — eliminates separate norm dispatch + barrier.
//
// Combines two Q8_0 matrix-vector multiplies that share the same input vector
// (e.g. SSM qkv 8192x2048 + gate 4096x2048) with inline RMSNorm computation.
// Each simdgroup reads hidden and norm_weight from L1 cache, computes the
// normalization factor independently, and uses the normed input directly for
// the DMMV — no intermediate norm_buf write or barrier needed.

kernel void main0(
    constant DualQ8DmmvPush& p [[buffer(0)]],
    device const uchar* W0 [[buffer(1)]],
    device const uchar* W1 [[buffer(2)]],
    device const float* hidden [[buffer(3)]],
    device float* Y0 [[buffer(4)]],
    device float* Y1 [[buffer(5)]],
    device const float* norm_weight [[buffer(6)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    device const float* h = hidden + (p.x_offset >> 2);

    // Step 1: Compute RMS normalization factor from raw hidden state.
    float sq_sum = 0.0f;
    for (uint i = lane; i < p.K; i += 32u) {
        const float v = h[i];
        sq_sum += v * v;
    }
    sq_sum = simd_sum(sq_sum);
    const float rms_inv = rsqrt(sq_sum / float(p.K) + 1e-6f);

    const uint linear_row = tg_id * simdgroups_per_tg + sg_idx;
    const uint total_rows = p.M0 + p.M1;
    if (linear_row >= total_rows) return;

    const bool first = linear_row < p.M0;
    const uint row = first ? linear_row : (linear_row - p.M0);
    device const uchar* weights = first ? W0 : W1;
    device float* output = first ? (Y0 + (p.y0_offset >> 2)) : (Y1 + (p.y1_offset >> 2));
    const uint a_offset = first ? p.a0_offset : p.a1_offset;

    // Step 2: DMMV with inline-normalized input.
    const uint blocks_per_row = p.K >> 5;
    device const uchar* row_ptr = weights + a_offset + ulong(row) * ulong(blocks_per_row) * 34ull;

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
