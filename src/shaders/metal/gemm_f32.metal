#include <metal_stdlib>
using namespace metal;

// F32 batched GEMV for small projection matrices in Metal prefill.
//
// This intentionally keeps a simple one-row, one-token workgroup shape. It is
// used to unblock Gemma's F32 router projection inside the batched prefill
// path; the large weight-traffic wins remain in Q8_0/Q4_K/Q5_1 projections.

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

kernel void main0(
    constant GemmPush & args [[buffer(0)]],
    device const char * src0 [[buffer(1)]],
    device const char * src1 [[buffer(2)]],
    device       char * dst  [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint row = tgpig.x;
    const uint token = tgpig.y;
    if (row >= uint(args.ne0) || token >= uint(args.ne1)) return;

    device const float * w = (device const float *)(src0 + args.src0_off + args.nb01 * row);
    device const float * x = (device const float *)(src1 + args.nb11 * token);
    float acc = 0.0f;
    for (uint k = lane; k < uint(args.ne00); k += 32u) {
        acc = fma(w[k], x[k], acc);
    }

    const float sum = simd_sum(acc);
    if (lane == 0) {
        device float * out = (device float *) dst;
        out[token * uint(args.ne0) + row] = sum;
    }
}
