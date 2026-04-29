#include <metal_stdlib>
using namespace metal;

// Q8_0 GEMM kernel for Metal batched prefill.
//
// Computes dst[N x M] = src0[M x K] * src1[N x K], with Q8_0 weights and
// f32 activations. The tile shape mirrors gemm_q4k/gemm_q6k so the host can
// route Q8_0 projections through the same batched prefill orchestration.

struct GemmPush {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint32_t src0_off;
};

struct block_q8_0 {
    half d;
    char qs[32];
};

#define QK_Q8_0 32
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant constexpr int NR0 = 64;
constant constexpr int NR1 = 32;
constant constexpr int NK  = 32;
constant constexpr int NL0 = NK / 16;
constant constexpr int NL1 = NK / 8;

static void dequantize_q8_0(device const block_q8_0 * xb, short il, thread half4x4 & reg) {
    const float d = float(xb->d);
    device const char * q = xb->qs + 16 * il;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = half(d * float(q[i]));
    }
}

kernel void main0(
    constant GemmPush & args [[buffer(0)]],
    device const char * src0 [[buffer(1)]],
    device const char * src1 [[buffer(2)]],
    device       char * dst  [[buffer(3)]],
    threadgroup  char * shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = min(args.ne0 - r0, NR0);
    const short nr1 = min(args.ne1 - r1, NR1);

    const short lr0 = min((short)(tiitg / NL0), (short)(nr0 - 1));
    const short lr1 = min((short)(tiitg / NL1), (short)(nr1 - 1));
    const short il0 = tiitg % NL0;
    const short iy = 8 * (tiitg % NL1);

    device const float * y = (device const float *)(src1 + args.nb11 * (r1 + lr1) + args.nb10 * iy);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        device const block_q8_0 * x = (device const block_q8_0 *)(src0 + args.src0_off + args.nb01 * (r0 + lr0) + ulong(loop_k / QK_Q8_0) * 34ul);

        half4x4 temp_a;
        dequantize_q8_0(x, il0, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }

        {
            const short sx = tiitg % NL1;
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) = (half2x4)(*((device float2x4 *) y));
        }

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1) {
        device float * C = (device float *) dst +
            (r0 + 32 * (sgitg & 1)) +
            (r1 + 16 * (sgitg >> 1)) * args.ne0;
        FOR_UNROLL (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * args.ne0 * (i / 4), args.ne0, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float *) dst + r0 + (r1 + j) * args.ne0;
                device float4 * D4 = (device float4 *) D;
                threadgroup float  * C  = temp_str + j * NR0;
                threadgroup float4 * C4 = (threadgroup float4 *) C;
                int i = 0;
                for (; i < nr0 / 4; i++) {
                    *(D4 + i) = *(C4 + i);
                }
                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}
