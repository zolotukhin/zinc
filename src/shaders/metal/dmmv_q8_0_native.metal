#include <metal_stdlib>
using namespace metal;

// Native Metal Q8_0 DMMV — one thread per output row.
// Q8_0 block: 32 elements, 34 bytes
//   [0..1]  d   (float16) — scale
//   [2..33] qs  (32 bytes) — signed 8-bit quantized values

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
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.M) return;

    const uint nb = p.K / 32;
    const uint bpb = 34;
    device const uchar* src = W + p.a_offset + gid * nb * bpb;
    device const float* x = X + (p.x_offset / 4);

    float sum = 0.0f;

    for (uint b = 0; b < nb; b++) {
        device const uchar* block = src + b * bpb;
        const half d = *((device const half*)block);
        device const char* qs = (device const char*)(block + 2);
        const uint base = b * 32;

        float block_sum = 0.0f;
        for (uint j = 0; j < 32; j++) {
            block_sum += float(qs[j]) * x[base + j];
        }
        sum += float(d) * block_sum;
    }

    Y[(p.y_offset / 4) + gid] = sum;
}
