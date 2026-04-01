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

inline float s8_to_f32(uint x) {
    return float((x < 128u) ? int(x) : (int(x) - 256));
}

// Reuse the staged expert input vector across 8 rows so mixed q6_k experts
// can stay on the shared-command decode path without thrashing device memory.
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
    const uint row_offset = p.a_offset + expert_id * p.expert_stride + row * blocks_per_row * 210u;
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
                const uint bb = row_offset + (tile_block + bi) * 210u;
                const float d = fp16_to_fp32(uint(W[bb + 208u]) | (uint(W[bb + 209u]) << 8u));
                const uint tile_base = bi * 256u;

                for (uint g = 0u; g < 2u; g++) {
                    const uint qs_lo_base = bb + g * 64u;
                    const uint qs_hi_base = bb + 128u + g * 32u;
                    const uint scale_base = bb + 192u + g * 8u;
                    const uint scale_group = lane / 16u;

                    const uint ql0 = uint(W[qs_lo_base + lane]);
                    const uint ql1 = uint(W[qs_lo_base + 32u + lane]);
                    const uint qh = uint(W[qs_hi_base + lane]);
                    const float d_sc0 = d * s8_to_f32(uint(W[scale_base + scale_group]));
                    const float d_sc1 = d * s8_to_f32(uint(W[scale_base + 2u + scale_group]));
                    const float d_sc2 = d * s8_to_f32(uint(W[scale_base + 4u + scale_group]));
                    const float d_sc3 = d * s8_to_f32(uint(W[scale_base + 6u + scale_group]));

                    const float q0 = float((ql0 & 0x0Fu) | ((qh & 0x03u) << 4u)) - 32.0f;
                    const float q1 = float((ql1 & 0x0Fu) | (((qh >> 2u) & 0x03u) << 4u)) - 32.0f;
                    const float q2 = float((ql0 >> 4u) | (((qh >> 4u) & 0x03u) << 4u)) - 32.0f;
                    const float q3 = float((ql1 >> 4u) | (((qh >> 6u) & 0x03u) << 4u)) - 32.0f;

                    const uint base_col = tile_base + g * 128u + lane;
                    sum += (d_sc0 * q0) * x_cache[base_col];
                    sum += (d_sc1 * q1) * x_cache[base_col + 32u];
                    sum += (d_sc2 * q2) * x_cache[base_col + 64u];
                    sum += (d_sc3 * q3) * x_cache[base_col + 96u];
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
