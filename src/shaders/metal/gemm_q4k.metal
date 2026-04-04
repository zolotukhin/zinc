#include <metal_stdlib>
using namespace metal;

// Q4_K GEMM kernel — dequantize-then-multiply using simdgroup matrix operations.
// Port of llama.cpp kernel_mul_mm for Q4_K weights × f32 input → f32 output.
//
// Output tile: 64 rows × 32 columns per threadgroup (64 weight rows × 32 tokens)
// Threadgroup: 128 threads = 4 simdgroups × 32 threads
// K-tile: 32 elements per iteration
//
// Used during prefill to process multiple prompt tokens simultaneously.
// The simdgroup_multiply_accumulate operations provide different numerical
// characteristics than the DMMV kernel, giving stable output for models
// without Q/K norms (e.g. LLaMA).

struct GemmPush {
    int32_t  ne00;      // K dimension (hidden_dim / cols of weight matrix)
    int32_t  ne02;      // batch dim 0
    uint64_t nb01;      // row stride of weight matrix (bytes)
    uint64_t nb02;      // batch stride 0
    int32_t  ne12;      // batch dim 1
    uint64_t nb10;      // element stride of input (bytes, typically 4 for f32)
    uint64_t nb11;      // row stride of input (bytes, = ne00 * 4 for contiguous)
    uint64_t nb12;      // batch stride 1
    int32_t  ne0;       // M dimension (rows of weight / output dim)
    int32_t  ne1;       // N dimension (number of input vectors / tokens)
};

// Q4_K block: 144 bytes, 256 elements
struct block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

#define QK_K  256
#define QK_NL 16    // number of dequant sub-blocks per Q4_K block

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                           uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

static void dequantize_q4_K(device const block_q4_K * xb, short il, thread half4x4 & reg) {
    device const uchar * q = xb->qs;
    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? float(xb->d) : float(xb->d) / 16.f;
    const float mn  = float(xb->dmin);
    const float dl  = d * sc[0];
    const float ml  = mn * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = half(dl * (q[i] & mask) - ml);
    }
}

// Tile dimensions
constant constexpr int NR0 = 64;  // output rows per threadgroup
constant constexpr int NR1 = 32;  // output cols per threadgroup
constant constexpr int NK  = 32;  // K-dimension per iteration
constant constexpr int NL0 = NK/16; // = 2 (sub-blocks per thread for weight loading)
constant constexpr int NL1 = NK/8;  // = 4 (elements per thread for input loading)

kernel void main0(
    constant GemmPush & args [[buffer(0)]],
    device const char * src0 [[buffer(1)]],    // Q4_K weight matrix
    device const char * src1 [[buffer(2)]],    // f32 input matrix [ne1 × ne00]
    device       char * dst  [[buffer(3)]],    // f32 output matrix [ne1 × ne0]
    threadgroup  char * shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half * sa = (threadgroup half *)(shmem);           // 64×32 half = 4096 bytes
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);    // 32×32 half = 2048 bytes (+pad)

    const int r0 = tgpig.y * NR0;   // first output row
    const int r1 = tgpig.x * NR1;   // first output col (token index)

    const short nr0 = min(args.ne0 - r0, NR0);
    const short nr1 = min(args.ne1 - r1, NR1);

    // Thread's assigned row/col for cooperative loading
    const short lr0 = min((short)(tiitg / NL0), (short)(nr0 - 1));
    const short lr1 = min((short)(tiitg / NL1), (short)(nr1 - 1));

    const short il0 = (tiitg % NL0);
    short il = il0;

    const short offset1 = il0 / QK_NL;

    device const block_q4_K * x = (device const block_q4_K *)(src0 + args.nb01 * (r0 + lr0)) + offset1;

    const short iy = 8 * (tiitg % NL1);

    device const float * y = (device const float *)(src1 + args.nb11 * (r1 + lr1) + args.nb10 * iy);

    // Initialize 8 simdgroup accumulator matrices (4 A-tiles × 2 B-tiles = 8 combinations)
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    // Iterate over K dimension in steps of 32
    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // --- Load A (weights): dequantize Q4_K → half, store to threadgroup memory ---
        half4x4 temp_a;
        dequantize_q4_K(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;

            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i/4][i%4];
        }

        // --- Load B (input): float → half, store to threadgroup memory ---
        {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;

            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) = (half2x4)(*((device float2x4 *) y));
        }

        // Advance weight block pointer
        il = (il + 2 < QK_NL) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + QK_NL - 1) / QK_NL : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Multiply-accumulate using simdgroup 8×8 matrix ops ---
        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb = (sb + 2 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    // --- Store output ---
    if (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1) {
        // Fast path: full tile, write directly to device memory
        device float * C = (device float *) dst +
            (r0 + 32 * (sgitg & 1)) +
            (r1 + 16 * (sgitg >> 1)) * args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * args.ne0 * (i / 4), args.ne0, 0, false);
        }
    } else {
        // Slow path: partial tile at matrix boundary
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
