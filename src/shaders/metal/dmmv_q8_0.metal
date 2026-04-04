#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Native Metal Q8_0 DMMV — barrier-free, L1-cached X reads.
//
// Q8_0 is the dominant path for router, shared-expert, and SSM projections on
// the target Qwen3.5-35B-A3B model (74.5% of all DMMV data).
//
// Each simdgroup (32 lanes) handles one output row independently.  X is read
// directly from device memory and cached in L1 (X is at most 16 KiB for
// K<=4096, well within the per-core L1).  No threadgroup memory or barrier
// is used, making all simdgroups fully independent and maximizing memory
// pipeline utilization on Apple Silicon.
//
// Q8_0 block layout (34 bytes, 32 elements):
//   [0..1]   d  (float16) scale
//   [2..33]  qs (32 x int8)

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
            const half4 q_half = half4(q);
            const half4 x = half4(*(device const float4*)(input + x_base + (vi << 2)));
            acc = fma(scale, float(dot(q_half, x)), acc);
        }
    }

    const float sum = simd_sum(acc);
    if (lane == 0u) {
        output[row] = sum;
    }
}
