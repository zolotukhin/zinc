#include <metal_stdlib>
using namespace metal;

struct MoeDmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint expert_stride;
    uint x_expert_stride;
    uint x_offset;
    uint y_offset;
};

inline float fp16_to_fp32(uint h) {
    return float(as_type<half>(ushort(h)));
}

inline float2 get_scale_min_k5(uint j, device const uchar* scales) {
    if (j < 4u) {
        return float2(float(scales[j] & 63u), float(scales[4u + j] & 63u));
    }
    return float2(
        float((scales[4u + j] & 0x0Fu) | ((scales[j - 4u] >> 6u) << 4u)),
        float((scales[4u + j] >> 4u) | ((scales[j] >> 6u) << 4u))
    );
}

// Reuse the staged expert input vector across 8 rows so the mixed-quant MoE
// path does not fall back to the one-thread-per-row SPIRV-Cross shape.
#define TG_SIZE 256
#define ROWS_PER_TG (TG_SIZE / 32)
#define TILE_K 2048
#define TILE_BLOCKS (TILE_K / 256)

kernel void main0(
    device const uchar* W [[buffer(0)]],
    constant MoeDmmvPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    device const uint* expert_ids [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 local_pos [[thread_position_in_threadgroup]]
) {
    const uint expert_slot = tg_pos.y;
    const uint expert_id = expert_ids[expert_slot];
    device const float* input = X + (p.x_offset / 4u) + expert_slot * p.x_expert_stride;
    threadgroup float x_cache[TILE_K];

    const uint local_id = local_pos.x;
    const uint sg_idx = local_id / 32u;
    const uint lane = local_id % 32u;
    const uint row = tg_pos.x * ROWS_PER_TG + sg_idx;
    const bool row_active = row < p.M;

    const uint blocks_per_row = p.K / 256u;
    const uint row_offset = p.a_offset + expert_id * p.expert_stride + row * blocks_per_row * 176u;
    const uint y_base = (p.y_offset / 4u) + expert_slot * p.M;

    float sum = 0.0f;

    for (uint tile_block = 0u; tile_block < blocks_per_row; tile_block += TILE_BLOCKS) {
        const uint remaining_blocks = blocks_per_row - tile_block;
        const uint tile_blocks = (remaining_blocks < TILE_BLOCKS) ? remaining_blocks : TILE_BLOCKS;
        const uint tile_elems = tile_blocks * 256u;
        const uint x_tile_base = tile_block * 256u;

        for (uint i = local_id; i < tile_elems; i += TG_SIZE) {
            x_cache[i] = input[x_tile_base + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_active) {
            for (uint bi = 0u; bi < tile_blocks; bi++) {
                const uint bb = row_offset + (tile_block + bi) * 176u;
                const float d = fp16_to_fp32(uint(W[bb]) | (uint(W[bb + 1u]) << 8u));
                const float dmin = fp16_to_fp32(uint(W[bb + 2u]) | (uint(W[bb + 3u]) << 8u));
                device const uchar* scales = W + bb + 4u;
                device const uchar* high_bits = W + bb + 16u;
                device const uchar* quants = W + bb + 48u;
                const uint tile_base = bi * 256u;

                for (uint g = 0u; g < 4u; g++) {
                    const uint sb_lo = g * 2u;
                    const uint sb_hi = sb_lo + 1u;
                    const float2 sm_lo = get_scale_min_k5(sb_lo, scales);
                    const float2 sm_hi = get_scale_min_k5(sb_hi, scales);
                    const float factor_lo = d * sm_lo.x;
                    const float bias_lo = dmin * sm_lo.y;
                    const float factor_hi = d * sm_hi.x;
                    const float bias_hi = dmin * sm_hi.y;

                    const uint q_byte = uint(quants[g * 32u + lane]);
                    const uint qh_val = uint(high_bits[lane]);
                    const float v_lo = factor_lo * float((q_byte & 0x0Fu) | (((qh_val >> sb_lo) & 1u) << 4u)) - bias_lo;
                    const float v_hi = factor_hi * float((q_byte >> 4u) | (((qh_val >> sb_hi) & 1u) << 4u)) - bias_hi;
                    const uint col_lo = tile_base + g * 64u + lane;
                    const uint col_hi = col_lo + 32u;

                    sum += v_lo * x_cache[col_lo];
                    sum += v_hi * x_cache[col_hi];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row_active) {
        const float total = simd_sum(sum);
        if (lane == 0u) {
            Y[y_base + row] = total;
        }
    }
}
