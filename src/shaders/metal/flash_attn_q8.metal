#include <metal_stdlib>
using namespace metal;

struct FlashAttnPush {
    uint head_dim;
    uint n_heads;
    uint n_kv_heads;
    uint seq_len;
    uint page_size;
    uint kv_head_stride_bytes;
    uint kv_token_stride_bytes;
};

constant uint FLASH_TG_SIZE = 64;
constant uint FLASH_BLOCK_TOKENS = 256;
constant uint FLASH_MAX_HEAD_DIM = 256;
constant uint FLASH_MAX_HEAD_VEC4 = FLASH_MAX_HEAD_DIM / 4;

inline float reduceThreadgroupMax(
    float local_value,
    threadgroup float* scratch,
    uint tid,
    uint subgroup_size,
    uint simd_lane,
    uint simd_group
) {
    const float wave_max = simd_max(local_value);
    if (subgroup_size < FLASH_TG_SIZE) {
        if (simd_lane == 0u) {
            scratch[simd_group] = wave_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            const uint n_groups = (FLASH_TG_SIZE + subgroup_size - 1u) / subgroup_size;
            float merged = -INFINITY;
            for (uint sg = 0u; sg < n_groups; ++sg) {
                merged = fast::max(merged, scratch[sg]);
            }
            scratch[0] = merged;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return scratch[0];
    }
    return wave_max;
}

inline float reduceThreadgroupSum(
    float local_value,
    threadgroup float* scratch,
    uint tid,
    uint subgroup_size,
    uint simd_lane,
    uint simd_group
) {
    const float wave_sum = simd_sum(local_value);
    if (subgroup_size < FLASH_TG_SIZE) {
        if (simd_lane == 0u) {
            scratch[simd_group] = wave_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            const uint n_groups = (FLASH_TG_SIZE + subgroup_size - 1u) / subgroup_size;
            float merged = 0.0f;
            for (uint sg = 0u; sg < n_groups; ++sg) {
                merged += scratch[sg];
            }
            scratch[0] = merged;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return scratch[0];
    }
    return wave_sum;
}

inline uint kvBaseForTokenBytes(
    device const uint* page_table,
    constant FlashAttnPush& p,
    uint kv_head,
    uint token_idx
) {
    const uint page_size = max(p.page_size, 1u);
    const uint page = token_idx / page_size;
    const uint page_offset = token_idx % page_size;
    const uint physical_token = page_table[page] * page_size + page_offset;
    return physical_token * p.kv_token_stride_bytes + kv_head * p.kv_head_stride_bytes;
}

inline float4 loadQ8_0Vec4(device const uchar* base, uint vec4_idx) {
    const uint block_idx = vec4_idx >> 3u;
    const uint vec4_in_block = vec4_idx & 7u;
    device const uchar* block = base + block_idx * 34u;
    const float scale = float(as_type<half>(*(device const ushort*)(block)));
    device const packed_char4* quants = (device const packed_char4*)(block + 2u);
    const char4 q = char4(quants[vec4_in_block]);
    return float4(float(q[0]), float(q[1]), float(q[2]), float(q[3])) * scale;
}

kernel void main0(
    constant FlashAttnPush& p [[buffer(0)]],
    device const uint* page_table [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device const uchar* k_cache [[buffer(3)]],
    device const uchar* v_cache [[buffer(4)]],
    device float* out [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint subgroup_size [[thread_execution_width]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint q_per_kv = max(p.n_heads / max(p.n_kv_heads, 1u), 1u);
    const uint kv_head = head / q_per_kv;
    const uint q_base = head * p.head_dim;
    const uint vec4_dim = p.head_dim >> 2;
    const float scale = rsqrt((float)p.head_dim);
    const bool contiguous_kv = p.page_size == 0u;

    threadgroup float4 q_cache4[FLASH_MAX_HEAD_VEC4];
    threadgroup float4 acc_cache4[FLASH_MAX_HEAD_VEC4];
    threadgroup float scores[FLASH_BLOCK_TOKENS];
    threadgroup float reduce[FLASH_TG_SIZE];
    threadgroup float running_max;
    threadgroup float running_sum;

    if (tid < vec4_dim) {
        q_cache4[tid] = *(device const float4*)(q + q_base + (tid << 2));
        acc_cache4[tid] = float4(0.0f);
    }
    if (tid == 0u) {
        running_max = -INFINITY;
        running_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint block_start = 0; block_start < p.seq_len; block_start += FLASH_BLOCK_TOKENS) {
        const uint block_tokens = min(FLASH_BLOCK_TOKENS, p.seq_len - block_start);
        const uint block_base = block_start * p.kv_token_stride_bytes + kv_head * p.kv_head_stride_bytes;
        float local_max = -INFINITY;

        for (uint token_offset = tid; token_offset < block_tokens; token_offset += FLASH_TG_SIZE) {
            const uint token_idx = block_start + token_offset;
            const uint kv_base = contiguous_kv
                ? (block_base + token_offset * p.kv_token_stride_bytes)
                : kvBaseForTokenBytes(page_table, p, kv_head, token_idx);

            float score = 0.0f;
            for (uint i = 0; i < vec4_dim; ++i) {
                score += dot(q_cache4[i], loadQ8_0Vec4(k_cache + kv_base, i));
            }
            score *= scale;
            scores[token_offset] = score;
            local_max = fast::max(local_max, score);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float block_max = reduceThreadgroupMax(local_max, reduce, tid, subgroup_size, simd_lane, simd_group);
        const float next_max = fast::max(running_max, block_max);

        float local_sum = 0.0f;
        for (uint token_offset = tid; token_offset < block_tokens; token_offset += FLASH_TG_SIZE) {
            const float weight = fast::exp(scores[token_offset] - next_max);
            scores[token_offset] = weight;
            local_sum += weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float block_sum = reduceThreadgroupSum(local_sum, reduce, tid, subgroup_size, simd_lane, simd_group);
        const float rescale = running_sum > 0.0f ? fast::exp(running_max - next_max) : 0.0f;

        if (tid < vec4_dim) {
            float4 acc = acc_cache4[tid] * rescale;

            if (contiguous_kv) {
                uint kv_base = block_base;
                for (uint token_offset = 0; token_offset < block_tokens; ++token_offset) {
                    acc += loadQ8_0Vec4(v_cache + kv_base, tid) * scores[token_offset];
                    kv_base += p.kv_token_stride_bytes;
                }
            } else {
                for (uint token_offset = 0; token_offset < block_tokens; ++token_offset) {
                    const uint kv_base = kvBaseForTokenBytes(page_table, p, kv_head, block_start + token_offset);
                    acc += loadQ8_0Vec4(v_cache + kv_base, tid) * scores[token_offset];
                }
            }

            acc_cache4[tid] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            running_sum = running_sum * rescale + block_sum;
            running_max = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    if (tid < vec4_dim) {
        *(device float4*)(out + q_base + (tid << 2)) = acc_cache4[tid] * inv_sum;
    }
}
