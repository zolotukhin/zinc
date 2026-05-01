#include <metal_stdlib>
using namespace metal;

// Dense Gemma single-token gate/up Q4_K matvec fused with GeGLU.
//
// The row layout follows llama.cpp's kernel_mul_mv_q4_K_f32:
// 64 threads per threadgroup, 2 simdgroups, 2 rows per simdgroup.
// Unlike the older K=5376-specialized kernels, this does not stage the input
// vector in threadgroup memory. It only fuses the two same-shape projections
// and the activation so the input lanes are loaded once for gate+up.

struct DualQ4KDmmvPush {
    uint M0;
    uint M1;
    uint K;
    uint a0_offset;
    uint a1_offset;
    uint x_offset;
    uint y0_offset;
    uint y1_offset;
};

#define NSG 2
#define NR0 2
#define QK_K 256
#define BLOCK_SIZE 144
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

inline float q4k_block_dot(
    device const uchar* block,
    thread const float* yl,
    thread const float* yh,
    float4 sumy,
    ushort iq,
    ushort ir
) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    device const ushort* sc = (device const ushort*)(block + 4) + iq;
    device const ushort* q1 = (device const ushort*)(block + 16) + 16 * iq + 4 * ir;
    device const half* dh = (device const half*)block;

    sc16[0] = sc[0] & kmask1;
    sc16[1] = sc[2] & kmask1;
    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

    device const ushort* q2 = q1 + 32;

    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

    FOR_UNROLL (short i = 0; i < 4; ++i) {
        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
    }

    return dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
            (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
            (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
            (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
        dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
}

inline float geglu(float gate, float up) {
    const float g3 = gate * gate * gate;
    float inner = 0.7978845608f * (gate + 0.044715f * g3);
    inner = clamp(inner, -15.0f, 15.0f);
    const float gelu_gate = 0.5f * gate * (1.0f + precise::tanh(inner));
    return gelu_gate * up;
}

kernel void main0(
    device const uchar* W0 [[buffer(0)]],
    device const uchar* W1 [[buffer(1)]],
    constant DualQ4KDmmvPush& p [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* activatedY [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const int nb = p.K / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const int row_bytes = nb * BLOCK_SIZE;

    device const uchar* gate_src = W0 + p.a0_offset;
    device const uchar* up_src = W1 + p.a1_offset;
    device const float* x = X + (p.x_offset / 4);
    device float* out = activatedY + (p.y0_offset / 4);

    float yl[16];
    float yh[16];
    float gate_sum[NR0] = {0.f, 0.f};
    float up_sum[NR0] = {0.f, 0.f};

    device const float* y4 = x + ix * QK_K + 64 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        FOR_UNROLL (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0];     sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32];    sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128];   sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160];   sumy[3] += yh[i + 8];
        }

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            const int dst_row = first_row + row;
            if (dst_row < int(p.M0)) {
                const ulong row_off = ulong(dst_row) * ulong(row_bytes) + ulong(ib) * BLOCK_SIZE;
                gate_sum[row] += q4k_block_dot(gate_src + row_off, yl, yh, sumy, iq, ir);
                up_sum[row] += q4k_block_dot(up_src + row_off, yl, yh, sumy, iq, ir);
            }
        }

        y4 += 4 * QK_K;
    }

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const int dst_row = first_row + row;
        if (dst_row < int(p.M0)) {
            const float gate_total = simd_sum(gate_sum[row]);
            const float up_total = simd_sum(up_sum[row]);
            if (tiisg == 0) {
                out[dst_row] = geglu(gate_total, up_total);
            }
        }
    }
}
