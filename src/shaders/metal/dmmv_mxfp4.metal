#include <metal_stdlib>
using namespace metal;

// MXFP4 DMMV kernel — one thread per output row.
// MXFP4 block: 32 elements, 17 bytes
//   [0]     e    (uint8) — E8M0 shared exponent
//   [1..16] qs   (16 bytes) — packed 4-bit E2M1 values (2 per byte)

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// E2M1 FP4 lookup table: maps 4-bit value to float
// Sign (1 bit) | Exponent (2 bits) | Mantissa (1 bit)
// Matches llama.cpp kvalues_mxfp4_f
constant float kvalues_mxfp4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// E8M0 to float: pure exponent, no mantissa
static inline float e8m0_to_fp32(uchar x) {
    uint bits;
    if (x == 0) {
        bits = 0x00400000u; // smallest non-zero
    } else {
        bits = uint(x) << 23;
    }
    return as_type<float>(bits);
}

kernel void main0(
    constant DmmvPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.M) return;

    const uint nb = p.K / 32;      // blocks per row
    const uint bpb = 17;            // bytes per MXFP4 block
    device const uchar* src = W + p.a_offset + gid * nb * bpb;
    device const float* x = X + (p.x_offset / 4);

    float sum = 0.0f;

    for (uint b = 0; b < nb; b++) {
        device const uchar* block = src + b * bpb;

        // E8M0 shared exponent
        const float d = e8m0_to_fp32(block[0]);

        // Packed 4-bit values (16 bytes = 32 elements)
        device const uchar* qs = block + 1;
        const uint base = b * 32;

        // Elements 0-15: low nibble of each byte
        // Elements 16-31: high nibble of each byte
        for (uint j = 0; j < 16; j++) {
            const uchar q_byte = qs[j];
            const float v_lo = d * kvalues_mxfp4[q_byte & 0x0F];
            const float v_hi = d * kvalues_mxfp4[q_byte >> 4];

            sum += v_lo * x[base + j]      +
                   v_hi * x[base + j + 16];
        }
    }

    Y[(p.y_offset / 4) + gid] = sum;
}
