#include <metal_stdlib>
using namespace metal;

// Exact F32 GEMM for small Gemma prefill projections such as router logits:
// weight[M,K] x input[N,K] -> output[N,M]. This intentionally avoids the
// half-tile simdgroup path because router top-k is sensitive to F32 ranking.

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
    uint3 gid [[thread_position_in_grid]]
) {
    const uint row = gid.x;
    const uint tok = gid.y;
    if (row >= uint(args.ne0) || tok >= uint(args.ne1)) {
        return;
    }

    device const float * w = (device const float *)(src0 + args.src0_off + row * args.nb01);
    device const float * x = (device const float *)(src1 + tok * args.nb11);

    float acc = 0.0f;
    for (uint k = 0; k < uint(args.ne00); ++k) {
        acc += w[k] * x[k];
    }

    device float * out = (device float *)dst;
    out[tok * uint(args.ne0) + row] = acc;
}
