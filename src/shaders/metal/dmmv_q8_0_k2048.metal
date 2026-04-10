#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// K <= 2048 specialization of Q8_0 DMMV — barrier-free, L1-cached X reads.
//
// Previous version staged X into threadgroup memory and synchronized all
// simdgroups with a threadgroup_barrier.  This couples simdgroups within a
// threadgroup: none can start computing until ALL finish loading X.  On the
// M4 Max (40 GPU cores, ~100-cycle memory latency) this limits independent
// memory request streams to 8 per core (one per resident threadgroup),
// under-saturating the memory pipeline and leaving bandwidth at ~19%.
//
// This version eliminates threadgroup memory entirely.  Each simdgroup reads
// X directly from device memory; after the first access on a core the 4-8 KiB
// vector lands in L1 cache and subsequent simdgroups get L1 hits.  All
// simdgroups become fully independent, increasing concurrent memory streams
// from 8 to 64+ per core and dramatically improving bandwidth utilization.
//
// Trade-off: each simdgroup re-reads X from L1 (~217 GB/s per core) instead
// of threadgroup memory.  For K=2048 the X vector is only 8 KiB — well within
// L1 — so the L1 reads complete far faster than device weight fetches.

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
    const uint row = tg_id * simdgroups_per_tg + sg_idx;
    if (row >= p.M) return;

    device const float* input = X + (p.x_offset >> 2);
    device float* output = Y + (p.y_offset >> 2);

    const uint blocks_per_row = p.K >> 5;
    device const uchar* row_ptr = W + p.a_offset + ulong(row) * ulong(blocks_per_row) * 34ull;

    float acc = 0.0f;
    for (uint bi = lane; bi < blocks_per_row; bi += 32u) {
        device const uchar* block = row_ptr + bi * 34u;
        const float scale = float(as_type<half>(*(device const ushort*)(block)));
        device const packed_char4* quants = (device const packed_char4*)(block + 2u);
        const uint x_base = bi << 5;  // bi * 32 elements

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const char4 q = char4(quants[vi]);
            const float4 q_f = float4(q);
            const float4 x_f = *(device const float4*)(input + x_base + (vi << 2));
            acc = fma(scale, dot(q_f, x_f), acc);
        }
    }

    const float sum = simd_sum(acc);
    if (lane == 0u) {
        output[row] = sum;
    }
}
