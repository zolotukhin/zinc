#include <metal_stdlib>
using namespace metal;

// Push constants for DMMV dispatch (matches Zig DmmvPush layout).
struct DmmvPush {
    uint M;        // rows
    uint K;        // cols (max 2048)
    uint a_offset; // byte offset into weight matrix
    uint x_offset; // byte offset into input vector
    uint y_offset; // byte offset into output vector
};

// Port of llama.cpp's kernel_mul_mv_q4_K_f32 (non-ext variant).
// K <= 2048 specialization — identical accumulation pattern to the general
// dmmv_q4k.metal, kept as a separate shader only so the pipeline name is distinct.
//
// Thread organization (matches llama.cpp with N_SG_Q4_K=2, N_R0_Q4_K=2):
//   64 threads per threadgroup = 2 simdgroups x 32 threads
//   Each simdgroup processes 2 rows => 4 rows per threadgroup

#define NSG   2
#define NR0   2
#define QK_K  256
#define BLOCK_SIZE 144

kernel void main0(
    device const uchar* W [[buffer(0)]],
    constant DmmvPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    const short ix = tiisg / 8;  // 0..3
    const short it = tiisg % 8;  // 0..7
    const short iq = it / 4;     // 0 or 1
    const short ir = it % 4;     // 0..3

    const int nb = p.K / QK_K;   // blocks per row

    const int r0 = tgpig.x;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const int nb01 = nb * BLOCK_SIZE;

    device const uchar* src0 = W + p.a_offset;
    device const float* src1 = X + (p.x_offset / 4);

    device const uchar* x_base = src0 + (uint64_t)first_row * nb01;
    device const float* y = src1;

    float yl[16];
    float yh[16];

    float sumf[NR0] = {0.f, 0.f};

    device const float* y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const ushort* sc = (device const ushort*)(x_base + (uint64_t)ib * BLOCK_SIZE + 4) + iq;
        device const ushort* q1 = (device const ushort*)(x_base + (uint64_t)ib * BLOCK_SIZE + 16) + 16 * iq + 4 * ir;
        device const half*   dh = (device const half*)(x_base + (uint64_t)ib * BLOCK_SIZE);

        for (short row = 0; row < NR0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const ushort* q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                   (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                   (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                   (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += nb01 / 2;
            sc += nb01 / 2;
            dh += nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    device float* dst_f32 = Y + (p.y_offset / 4);

    for (int row = 0; row < NR0 && first_row + row < (int)p.M; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}
