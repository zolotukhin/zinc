#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Native Metal Q5_K DMMV — simdgroup-per-row with threadgroup input cache.
//
// Replaces the SPIRV-Cross auto-compiled shader which has:
// - No threadgroup input caching (64× redundant device reads per WG)
// - Single thread per row (no SIMD cooperation)
// - Byte-at-a-time weight access
//
// Q5_K block layout (176 bytes, 256 elements):
//   [0..1]   d     (float16) scale
//   [2..3]   dmin  (float16) min scale
//   [4..15]  scales (12 bytes: 8 packed 6-bit sub-block scales/mins)
//   [16..47] qh    (32 bytes: high bit 4 for each of 256 elements)
//   [48..175] qs   (128 bytes: low 4 bits, packed as nibbles)

kernel void main0(
    constant DmmvPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    // Stage input as half4 — K≤4096 → max 1024 entries = 8 KiB threadgroup memory.
    threadgroup half4 x_cache4[1024];

    const uint tg_size = simdgroups_per_tg * 32u;
    device const float* input = X + (p.x_offset >> 2);
    device float* output = Y + (p.y_offset >> 2);

    const uint k_vec4 = p.K >> 2;
    for (uint i = local_id; i < k_vec4; i += tg_size) {
        x_cache4[i] = half4(*(device const float4*)(input + (i << 2)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tg_id * simdgroups_per_tg + sg_idx;
    if (row >= p.M) return;

    const uint blocks_per_row = p.K >> 8; // K / 256
    device const uchar* row_ptr = W + p.a_offset + ulong(row) * ulong(blocks_per_row) * 176ull;

    float acc = 0.0f;

    for (uint b = 0; b < blocks_per_row; b++) {
        device const uchar* block = row_ptr + b * 176u;

        // Read d and dmin (float16 each)
        const float d = float(as_type<half>(*(device const ushort*)(block)));
        const float dmin = float(as_type<half>(*(device const ushort*)(block + 2)));

        // Read qh byte once for this lane — used across all 4 groups
        const uint qh_val = uint(block[16u + lane]);

        // Process 4 groups of 64 elements each.
        // Each group: 32 low-nibble elements + 32 high-nibble elements.
        // All 32 lanes process one element position each.
        #pragma unroll
        for (uint g = 0u; g < 4u; g++) {
            const uint sb_lo = g * 2u;
            const uint sb_hi = g * 2u + 1u;

            // Decode 6-bit scales and mins for both sub-blocks
            uint sc_lo, m_lo, sc_hi, m_hi;
            if (sb_lo < 4u) {
                sc_lo = uint(block[4u + sb_lo]) & 63u;
                m_lo  = uint(block[4u + sb_lo + 4u]) & 63u;
            } else {
                sc_lo = (uint(block[4u + sb_lo + 4u]) & 0xFu) | ((uint(block[4u + sb_lo - 4u]) >> 6u) << 4u);
                m_lo  = (uint(block[4u + sb_lo + 4u]) >> 4u) | ((uint(block[4u + sb_lo]) >> 6u) << 4u);
            }
            if (sb_hi < 4u) {
                sc_hi = uint(block[4u + sb_hi]) & 63u;
                m_hi  = uint(block[4u + sb_hi + 4u]) & 63u;
            } else {
                sc_hi = (uint(block[4u + sb_hi + 4u]) & 0xFu) | ((uint(block[4u + sb_hi - 4u]) >> 6u) << 4u);
                m_hi  = (uint(block[4u + sb_hi + 4u]) >> 4u) | ((uint(block[4u + sb_hi]) >> 6u) << 4u);
            }

            const float factor_lo = d * float(sc_lo);
            const float bias_lo   = dmin * float(m_lo);
            const float factor_hi = d * float(sc_hi);
            const float bias_hi   = dmin * float(m_hi);

            // Each lane reads its qs byte for this group
            const uint qs_byte = uint(block[48u + g * 32u + lane]);
            const uint q_lo = qs_byte & 0xFu;
            const uint q_hi = qs_byte >> 4u;

            // Extract high bits for this group
            const uint hb_lo = (qh_val >> sb_lo) & 1u;
            const uint hb_hi = (qh_val >> sb_hi) & 1u;

            const float v_lo = float(q_lo | (hb_lo << 4u));
            const float v_hi = float(q_hi | (hb_hi << 4u));

            // Read cached input: block b has 256 elements = 64 half4 entries
            const uint x_base = b * 64u + g * 16u;
            const half xval_lo = x_cache4[x_base + (lane >> 2u)][lane & 3u];
            const half xval_hi = x_cache4[x_base + 8u + (lane >> 2u)][lane & 3u];

            acc += (factor_lo * v_lo - bias_lo) * float(xval_lo);
            acc += (factor_hi * v_hi - bias_hi) * float(xval_hi);
        }
    }

    const float sum = simd_sum(acc);
    if (lane == 0u) {
        output[row] = sum;
    }
}
