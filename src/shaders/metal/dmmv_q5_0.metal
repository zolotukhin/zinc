#include <metal_stdlib>
using namespace metal;

// Q5_0 DMMV kernel — ported from llama.cpp's optimized approach.
// Key optimization: factor out the 'd' scale and '-16' bias per block,
// reducing float32 multiplications from 64 to ~36 per block and
// improving accumulation precision.
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
        const float d = float(*((device const half*)block));

        // Read 5th-bit mask (uint32 at bytes 2-5) — use byte reads for safe alignment
        const uint qh = uint(block[2]) | (uint(block[3]) << 8) | (uint(block[4]) << 16) | (uint(block[5]) << 24);

        // Read lower nibbles (16 bytes at bytes 6-21)
        device const uchar* qs = block + 6;

        const uint base = b * 32;

        // Accumulate raw (unsigned) quant * input products and input sum separately.
        // This factors out the -16 bias: d * (sum_qx - 16 * sum_x)
        // Reduces per-element float muls and improves precision.
        float sum_qx = 0.0f;
        float sum_x = 0.0f;

        for (uint j = 0; j < 16; j++) {
            const uchar q_byte = qs[j];
            const uint lo = q_byte & 0x0F;
            const uint hi = q_byte >> 4;

            const uint bit_lo = (qh >> j)        & 1;
            const uint bit_hi = (qh >> (j + 16)) & 1;

            // Unsigned quant values (0-31)
            const uint q0 = lo | (bit_lo << 4);
            const uint q1 = hi | (bit_hi << 4);

            const float x0 = x[base + j];
            const float x1 = x[base + 16 + j];

            sum_qx += float(q0) * x0 + float(q1) * x1;
            sum_x  += x0 + x1;
        }

        // Apply scale and bias: d * (sum_qx - 16 * sum_x)
        sum += d * (sum_qx - 16.0f * sum_x);
    }

    // SIMD reduction across 32 threads
    sum = simd_sum(sum);

    // Thread 0 writes the result
    if (tid == 0) {
        Y[(p.y_offset / 4) + row] = sum;
    }
}
