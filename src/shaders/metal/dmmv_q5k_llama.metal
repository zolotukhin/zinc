#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Port of the reference kernel_mul_mv_q5_K_f32_impl (N_R0_Q5_K=1) for dense
// single-token decode, in ZINC's buffer/push convention (W[0], p[1], X[2],
// Y[3]) — mirrors dmmv_q6k_llama.metal.
//
// Replaces dmmv_q5k_native (simdgroup-per-row with a 16 KiB threadgroup input
// cache + byte-wise weight reads), which measured only ~190 GB/s (34% of M4
// Max peak) on the huge Q5_K lm-head (M=202048, K=6656). llama's structure —
// 4-way lane split over the 256-elem block (ix = tiisg%4), register-cached
// input with a sumy dmin-correction, and packed uchar weight reads — is the
// same shape as the Q6_K llama port that hits ~465 GB/s.
//
// Q5_K block (176 bytes, 256 elems): d(half)@0, dmin(half)@2, scales[12]@4,
// qh[32]@16, qs[128]@48.
#ifndef ZINC_Q5K_NSG
#define ZINC_Q5K_NSG 2
#endif

#define NSG ZINC_Q5K_NSG
#define QK_K 256
#define BLOCK_SIZE 176
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

kernel void main0(
    device const uchar* W [[buffer(0)]],
    constant DmmvPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint nb = p.K / QK_K;
    const uint row_bytes = nb * BLOCK_SIZE;
    const uint first_row = tgpig.x * NSG + uint(sgitg); // NR0 = 1
    // Uniform per-simdgroup guard (first_row is invariant across the simdgroup),
    // so a tail simdgroup never reads out-of-bounds weight rows for odd M.
    if (first_row >= p.M) return;

    device const uchar* src0 = W + p.a_offset;
    device const float* yy = X + (p.x_offset / 4u);

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short tid = tiisg / 4u;
    const short ix  = tiisg % 4u;
    const short iq  = tid / 4u;
    const short ir  = tid % 4u;

    const short l0 = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    const uint8_t hm1 = 1u << (2 * iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t* sc8 = (thread const uint8_t*)sc16;

    // Row base and the per-block field pointers (byte offsets into the block).
    device const uchar* row_ptr = src0 + ulong(first_row) * ulong(row_bytes);
    device const float* y1 = yy + ix * QK_K + y_offset;

    float yl[16], yh[16];
    float sumf = 0.0f;

    for (uint i = ix; i < nb; i += 4u) {
        device const uchar* block = row_ptr + i * BLOCK_SIZE;
        device const uint8_t* q1 = block + 48u + uint(q_offset);       // qs + q_offset
        device const uint8_t* qh = block + 16u + uint(l0);             // qh + l0
        device const half*    dh = (device const half*)(block);       // &d  (d, dmin)
        device const uint16_t* a = (device const uint16_t*)(block + 4u) + iq; // scales, uint16 view

        device const float* y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        device const uint8_t* q2 = q1 + 64;

        sc16[0] = a[0] & kmask1;
        sc16[1] = a[2] & kmask1;
        sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
        sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

        float4 acc1 = {0.f};
        float4 acc2 = {0.f};
        FOR_UNROLL (short l = 0; l < 8; ++l) {
            uint8_t h = qh[l];
            acc1[0] += yl[l+0] * (q1[l] & 0x0F);
            acc1[1] += yl[l+8] * (q1[l] & 0xF0);
            acc1[2] += yh[l+0] * (q2[l] & 0x0F);
            acc1[3] += yh[l+8] * (q2[l] & 0xF0);
            acc2[0] += h & hm1 ? yl[l+0] : 0.f;
            acc2[1] += h & hm2 ? yl[l+8] : 0.f;
            acc2[2] += h & hm3 ? yh[l+0] : 0.f;
            acc2[3] += h & hm4 ? yh[l+8] : 0.f;
        }

        const float d = float(dh[0]);
        const float dmin = float(dh[1]);
        sumf += d * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                     sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                     sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                     sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

        y1 += 4 * QK_K;
    }

    device float* out = Y + (p.y_offset / 4u);
    const float tot = simd_sum(sumf);
    if (tiisg == 0u && first_row < p.M) {
        out[first_row] = tot;
    }
}
