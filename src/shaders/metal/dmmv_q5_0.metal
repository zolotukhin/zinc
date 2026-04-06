#include <metal_stdlib>
using namespace metal;

// Q5_0 DMMV kernel — multiple threads per row with SIMD reduction.
// Uses 32 threads per row (1 simdgroup), each thread handles a subset of blocks.
// This dramatically improves float32 accumulation precision vs single-thread.
//
// Q5_0 block: 32 elements, 22 bytes
//   [0..1]  d    (float16) — scale
//   [2..5]  qh   (4 bytes) — 5th bit for each of 32 elements
//   [6..21] qs   (16 bytes) — lower 4 bits, packed as nibbles

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

kernel void main0(
    constant DmmvPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]
) {
    // 32 threads per simdgroup, 2 simdgroups per threadgroup = 2 rows per workgroup
    const uint row = tgid * 2 + sgid;
    if (row >= p.M) return;

    const uint nb = p.K / 32;      // blocks per row
    const uint bpb = 22;            // bytes per Q5_0 block
    device const uchar* src = W + p.a_offset + row * nb * bpb;
    device const float* x = X + (p.x_offset / 4);

    float sum = 0.0f;

    // Each of 32 threads handles blocks [tid, tid+32, tid+64, ...]
    for (uint b = tid; b < nb; b += 32) {
        device const uchar* block = src + b * bpb;

        // Read scale (fp16 at bytes 0-1)
        const half d = *((device const half*)block);

        // Read 5th-bit mask (uint32 at bytes 2-5)
        const uint qh = *((device const uint*)&block[2]);

        // Read lower nibbles (16 bytes at bytes 6-21)
        device const uchar* qs = block + 6;

        const uint base = b * 32;

        // Accumulate all 32 elements of this block
        float block_sum = 0.0f;
        for (uint j = 0; j < 16; j++) {
            const uchar q_byte = qs[j];
            const uint lo = q_byte & 0x0F;
            const uint hi = q_byte >> 4;

            const uint bit_lo = (qh >> j)        & 1;
            const uint bit_hi = (qh >> (j + 16)) & 1;

            const float v0 = float(d) * float(int(lo | (bit_lo << 4)) - 16);
            const float v1 = float(d) * float(int(hi | (bit_hi << 4)) - 16);

            block_sum += v0 * x[base + j]      +
                         v1 * x[base + 16 + j];
        }
        sum += block_sum;
    }

    // SIMD reduction across 32 threads
    sum = simd_sum(sum);

    // Thread 0 writes the result
    if (tid == 0) {
        Y[(p.y_offset / 4) + row] = sum;
    }
}
