#include <metal_stdlib>
using namespace metal;

// Push constants for DMMV dispatch (matches Zig DmmvPush layout).
struct DmmvPush {
    uint M;        // rows
    uint K;        // cols (max 4096)
    uint a_offset; // byte offset into weight matrix
    uint x_offset; // byte offset into input vector
    uint y_offset; // byte offset into output vector
};

// Q4_K scale extraction — matches GGML Q4_K format.
// Returns (scale, min) for sub-block j (0..7).
inline float2 get_scale_min_k4(uint j, device const uchar* sc) {
    if (j < 4) {
        return float2(float(sc[j] & 63), float(sc[j + 4] & 63));
    }
    return float2(
        float((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4)),
        float(((sc[j + 4] >> 4) & 0xF) | ((sc[j] >> 6) << 4))
    );
}

// Native Metal Q4_K dequant matrix-vector multiply.
//
// One simdgroup (32 threads) processes one row. Adjacent threads read adjacent
// quant bytes for coalesced device memory access. simd_sum reduces the partial
// dot products — no threadgroup reduction barrier needed.
//
// Threadgroup: 256 threads = 8 simdgroups = 8 rows per threadgroup.
// The input vector is staged once into threadgroup memory and then reused by
// all 8 rows, avoiding repeated device reads of the same 8-16 KB vector.
//
// Q4_K block layout (144 bytes, 256 elements):
//   [0..1]   d    (float16)  — super-block scale
//   [2..3]   dmin (float16)  — super-block min
//   [4..15]  scales (12 B)   — 8 sub-block scale/min pairs, packed 6-bit
//   [16..143] quants (128 B) — 256 4-bit values: 4 groups of 32 bytes,
//             low nibble = first 32 elements, high nibble = next 32

#define TG_SIZE 256
#define ROWS_PER_TG (TG_SIZE / 32)

kernel void main0(
    device const uchar* W [[buffer(0)]],
    constant DmmvPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    device const float* input = X + (p.x_offset / 4);
    threadgroup float4 x_cache4[1024];

    // Q4_K rows are multiples of 256 elements, so all current K values are
    // 16-byte aligned and can be staged as float4 chunks.
    const uint k_vec4 = p.K >> 2;
    for (uint i = local_id; i < k_vec4; i += TG_SIZE) {
        x_cache4[i] = *(device const float4*)(input + (i << 2));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = tg_id * ROWS_PER_TG + sg_idx;
    if (row >= p.M) return;

    uint bpr = p.K / 256;  // Q4_K blocks per row
    device const uchar* row_ptr = W + p.a_offset + ulong(row) * ulong(bpr) * 144;

    float acc = 0.0f;

    for (uint bi = 0; bi < bpr; bi++) {
        device const uchar* block = row_ptr + bi * 144;

        // Super-block scale and min (f16 → f32)
        float d    = float(as_type<half>(*(device const ushort*)(block)));
        float dmin = float(as_type<half>(*(device const ushort*)(block + 2)));

        device const uchar* scales = block + 4;
        device const uchar* quants = block + 16;

        // Each lane reads 4 consecutive quant bytes (= 8 values: 4 low + 4 high nibble).
        // 32 lanes × 4 bytes = 128 bytes = full quant region → perfectly coalesced.
        uint byte_off  = lane * 4;
        uint j         = byte_off / 32;         // quarter index (0..3)
        uint local_off = byte_off % 32;         // offset within quarter

        // lane*4 keeps both quant and input accesses 16-byte aligned, so load
        // four packed quant bytes and four input floats at once.
        uchar4 qbytes = *(device const uchar4*)(quants + byte_off);

        // Sub-block scales: low nibble uses sub j*2, high nibble uses sub j*2+1
        float2 sm_lo = get_scale_min_k4(j * 2,     scales);
        float2 sm_hi = get_scale_min_k4(j * 2 + 1, scales);

        float d_sc_lo = d * sm_lo.x;
        float d_m_lo  = dmin * sm_lo.y;
        float d_sc_hi = d * sm_hi.x;
        float d_m_hi  = dmin * sm_hi.y;

        // Column indices in the full row
        uint col_lo = bi * 256 + j * 64 + local_off;
        uint col_hi = col_lo + 32;

        float4 x_lo = x_cache4[col_lo >> 2];
        float4 x_hi = x_cache4[col_hi >> 2];

        uchar4 q_lo = uchar4(
            qbytes.x & 0x0F,
            qbytes.y & 0x0F,
            qbytes.z & 0x0F,
            qbytes.w & 0x0F
        );
        uchar4 q_hi = uchar4(
            qbytes.x >> 4,
            qbytes.y >> 4,
            qbytes.z >> 4,
            qbytes.w >> 4
        );

        float4 lo_vals = fma(float4(q_lo), float4(d_sc_lo), float4(-d_m_lo));
        float4 hi_vals = fma(float4(q_hi), float4(d_sc_hi), float4(-d_m_hi));

        acc += dot(lo_vals, x_lo);
        acc += dot(hi_vals, x_hi);
    }

    // Reduce across simdgroup and write result
    float sum = simd_sum(acc);
    if (lane == 0) {
        Y[p.y_offset / 4 + row] = sum;
    }
}
