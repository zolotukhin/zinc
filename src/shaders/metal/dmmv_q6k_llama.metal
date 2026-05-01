#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

inline float fp16_to_fp32(uint h) {
    return float(as_type<half>(ushort(h)));
}

inline float s8_to_f32(uint x) {
    return float((x < 128u) ? int(x) : (int(x) - 256));
}

// Port of llama.cpp's kernel_mul_mv_q6_K_f32 for dense single-token decode.
// Thread organization matches llama.cpp N_SG_Q6_K=2, N_R0_Q6_K=2:
// 64 threads = 2 simdgroups, each simdgroup computes 2 rows.
#define NSG 2
#define NR0 2
#define QK_K 256
#define BLOCK_SIZE 210

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
    const uint first_row = (tgpig.x * NSG + uint(sgitg)) * NR0;
    const uint row_bytes = nb * BLOCK_SIZE;

    device const uchar* src0 = W + p.a_offset;
    device const float* src1 = X + (p.x_offset / 4u);

    constexpr uint kmask1 = 0x03u;
    constexpr uint kmask2 = 0x0Cu;
    constexpr uint kmask3 = 0x30u;
    constexpr uint kmask4 = 0xC0u;

    const ushort tid = tiisg / 2u;
    const ushort ix = tiisg % 2u;
    const ushort ip = tid / 8u;
    const ushort il = tid % 8u;
    const ushort l0 = 4u * il;
    const ushort is = 8u * ip + l0 / 16u;

    const uint y_offset = 128u * uint(ip) + uint(l0);
    const uint q_offset_l = 64u * uint(ip) + uint(l0);
    const uint q_offset_h = 32u * uint(ip) + uint(l0);

    float sumf[NR0] = {0.0f, 0.0f};

    for (uint bi = ix; bi < nb; bi += 2u) {
        device const float* y = src1 + bi * QK_K + y_offset;

        float yl[16];
        for (ushort l = 0u; l < 4u; ++l) {
            yl[4u * l + 0u] = y[l + 0u];
            yl[4u * l + 1u] = y[l + 32u];
            yl[4u * l + 2u] = y[l + 64u];
            yl[4u * l + 3u] = y[l + 96u];
        }

        for (ushort row = 0u; row < NR0; ++row) {
            const uint dst_row = first_row + uint(row);
            if (dst_row >= p.M) {
                continue;
            }

            device const uchar* block = src0 + ulong(dst_row) * ulong(row_bytes) + ulong(bi) * BLOCK_SIZE;
            device const uchar* q1 = block + q_offset_l;
            device const uchar* q2 = q1 + 32u;
            device const uchar* qh = block + 128u + q_offset_h;
            device const uchar* sc = block + 192u + uint(is);
            const float d = fp16_to_fp32(uint(block[208]) | (uint(block[209]) << 8u));

            float4 sums = float4(0.0f);
            for (ushort l = 0u; l < 4u; ++l) {
                const uint h = uint(qh[l]);
                const float q0 = float(int((uint(q1[l]) & 0x0Fu) | ((h & kmask1) << 4u)) - 32);
                const float q1v = float(int((uint(q2[l]) & 0x0Fu) | ((h & kmask2) << 2u)) - 32);
                const float q2v = float(int((uint(q1[l]) >> 4u) | ((h & kmask3) << 0u)) - 32);
                const float q3 = float(int((uint(q2[l]) >> 4u) | ((h & kmask4) >> 2u)) - 32);

                sums[0] += yl[4u * l + 0u] * q0;
                sums[1] += yl[4u * l + 1u] * q1v;
                sums[2] += yl[4u * l + 2u] * q2v;
                sums[3] += yl[4u * l + 3u] * q3;
            }

            sumf[row] += d * (
                sums[0] * s8_to_f32(uint(sc[0])) +
                sums[1] * s8_to_f32(uint(sc[2])) +
                sums[2] * s8_to_f32(uint(sc[4])) +
                sums[3] * s8_to_f32(uint(sc[6]))
            );
        }
    }

    device float* out = Y + (p.y_offset / 4u);
    for (ushort row = 0u; row < NR0; ++row) {
        const uint dst_row = first_row + uint(row);
        if (dst_row >= p.M) {
            continue;
        }

        const float total = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[dst_row] = total;
        }
    }
}
