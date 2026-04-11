#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Repacked Q8_0 DMMV — nr=2 multi-row, SIMD-coalesced weight reads.
//
// Standard Q8_0 blocks are 34 bytes each. When each SIMD lane reads a
// separate block, the stride-34 access pattern wastes ~88% of memory
// bandwidth on unused cache-line bytes. On M4 Max (546 GB/s), this
// limits effective bandwidth to ~85 GB/s.
//
// This kernel reads from a *repacked* layout where 32 consecutive Q8_0
// blocks are interleaved so that adjacent SIMD lanes read adjacent bytes:
//
//   Group (32 blocks = 1088 bytes):
//     [0..63]     32 × half scales  (lane L reads offset L*2)
//     [64..191]   chunk 0: 32 × char4 qs[0..3]   (lane L reads offset 64 + L*4)
//     [192..319]  chunk 1: 32 × char4 qs[4..7]    ...
//     ...
//     [960..1087] chunk 7: 32 × char4 qs[28..31]
//
// This achieves ~100% memory coalescing for weight reads.
//
// Requires: blocks_per_row is a multiple of 32 (K is a multiple of 1024).

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

    const uint blocks_per_row = p.K >> 5;            // K / 32
    const uint groups_per_row = blocks_per_row >> 5;  // blocks_per_row / 32
    const ulong group_bytes = 1088ull;                // 32 blocks * 34 bytes
    const ulong row_bytes = ulong(groups_per_row) * group_bytes;

    device const uchar* row0 = W + p.a_offset + ulong(base_row) * row_bytes;
    device const uchar* row1 = row0 + row_bytes;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint gi = 0u; gi < groups_per_row; ++gi) {
        device const uchar* g0 = row0 + ulong(gi) * group_bytes;
        device const uchar* g1 = row1 + ulong(gi) * group_bytes;

        // Coalesced scale read: 32 lanes × 2 bytes = 64 contiguous bytes
        const float s0 = float(as_type<half>(*(device const ushort*)(g0 + lane * 2u)));
        const float s1 = float(as_type<half>(*(device const ushort*)(g1 + lane * 2u)));

        // X base for this lane's block within the group
        const uint x_base = (gi * 32u + lane) << 5;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            // Coalesced weight read: 32 lanes × 4 bytes = 128 contiguous bytes
            const uint qo = 64u + vi * 128u + lane * 4u;
            const char4 q0 = as_type<char4>(*(device const int*)(g0 + qo));
            const char4 q1 = as_type<char4>(*(device const int*)(g1 + qo));

            const float4 x = *(device const float4*)(input + x_base + (vi << 2));
            acc0 = fma(s0, dot(float4(q0), x), acc0);
            acc1 = fma(s1, dot(float4(q1), x), acc1);
        }
    }

    const float sum0 = simd_sum(acc0);
    if (lane == 0u) output[base_row] = sum0;

    if (base_row + 1u < p.M) {
        const float sum1 = simd_sum(acc1);
        if (lane == 0u) output[base_row + 1u] = sum1;
    }
}
