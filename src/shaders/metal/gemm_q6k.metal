#include <metal_stdlib>
using namespace metal;

// Q6_K GEMM kernel — dequantize-then-multiply using simdgroup matrix operations.
// Port of llama.cpp kernel_mul_mm for Q6_K weights × f32 input → f32 output.
// Same tile structure as gemm_q4k.metal (64×32 output, 4 simdgroups, NK=32).

struct GemmPush {
    int32_t  ne00;      // K dimension
    int32_t  ne02;      // batch dim 0
    uint64_t nb01;      // row stride of weight matrix (bytes)
    uint64_t nb02;      // batch stride 0
    int32_t  ne12;      // batch dim 1
    uint64_t nb10;      // element stride of input (4 for f32)
    uint64_t nb11;      // row stride of input (ne00 * 4)
    uint64_t nb12;      // batch stride 1
    int32_t  ne0;       // M dimension (output rows)
    int32_t  ne1;       // N dimension (number of tokens)
    uint32_t src0_off;  // byte offset to weight data within the Metal buffer
};

// Q6_K block: 210 bytes, 256 elements
//   [0..127]   ql   (128 bytes, lower 4 bits of each element)
//   [128..191] qh   (64 bytes, upper 2 bits)
//   [192..207] scales (16 bytes, int8 scales)
//   [208..209] d    (float16, super-block scale)
struct block_q6_K {
    uchar ql[128];
    uchar qh[64];
    char scales[16];
    half d;
};

#define QK_K  256
#define QK_NL 16
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

static void dequantize_q6_K(device const block_q6_K * xb, short il, thread half4x4 & reg) {
    const half d_all = xb->d;
    device const ushort * ql = (device const ushort *)xb->ql;
    device const ushort * qh = (device const ushort *)xb->qh;
    device const char * scales = (device const char *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint kmask2 = il>1 ? 0xF0F0F0F0 : 0x0F0F0F0F;
    const float ml = float(d_all) * sc * 32.f;
    const float dl0 = float(d_all) * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uchar shr_h = il>2 ? 2 : 0;
    const uchar shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uchar shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint  low = (ql[2*i] | (uint)(ql[2*i+1] << 16)) & kmask2;
        const uint high = (qh[2*i] | (uint)(qh[2*i+1] << 16)) & kmask1;
        const uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = half(dl0 *  ((half)(q & 0xFF))       - ml);
        reg[i][1] = half(dl1 * ((float)(q & 0xFF00))     - ml);
        reg[i][2] = half(dl2 * ((float)(q & 0xFF0000))   - ml);
        reg[i][3] = half(dl3 * ((float)(q & 0xFF000000)) - ml);
    }
}

constant constexpr int NR0 = 64;
constant constexpr int NR1 = 32;
constant constexpr int NK  = 32;
constant constexpr int NL0 = NK/16;
constant constexpr int NL1 = NK/8;

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

    const short il0 = (tiitg % NL0);
    short il = il0;

    const short offset1 = il0 / QK_NL;

    device const block_q6_K * x = (device const block_q6_K *)(src0 + args.src0_off + args.nb01 * (r0 + lr0)) + offset1;

    const short iy = 8 * (tiitg % NL1);

    device const float * y = (device const float *)(src1 + args.nb11 * (r1 + lr1) + args.nb10 * iy);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        half4x4 temp_a;
        dequantize_q6_K(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i/4][i%4];
        }

        {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) = (half2x4)(*((device float2x4 *) y));
        }

        il = (il + 2 < QK_NL) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + QK_NL - 1) / QK_NL : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb = (sb + 2 * 64 * (sgitg / 2));

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
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1) {
        device float * C = (device float *) dst +
            (r0 + 32 * (sgitg & 1)) +
            (r1 + 16 * (sgitg >> 1)) * args.ne0;
        for (short i = 0; i < 8; i++) {
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
                threadgroup float  * C  = temp_str + (j * NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;
                int i = 0;
                for (; i < nr0 / 4; i++) *(D4 + i) = *(C4 + i);
                i *= 4;
                for (; i < nr0; i++) *(D + i) = *(C + i);
            }
        }
    }
}
