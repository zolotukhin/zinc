#include <metal_stdlib>
using namespace metal;

// Batched causal flash attention with Q8_0-quantized KV cache, for prefill.
//
// Mirrors flash_attn_batched.metal for the orchestration and softmax math,
// but reads K/V from Q8_0-encoded blocks instead of raw f32. Q8_0 block is
// 34 bytes: 2 bytes scale (half) + 32 × i8 quants. Each head_dim worth of
// KV at a token boundary is laid out as (head_dim / 32) contiguous blocks,
// so dim `d` is at block `d/32`, quant index `d%32`.
//
// Grid: (n_heads, n_queries, 1), threadgroup: 64 threads.

struct BatchedFlashAttnQ8Push {
    uint head_dim;
    uint n_heads;
    uint n_kv_heads;
    uint kv_len;
    uint n_queries;
    uint kv_pos_offset;
    uint kv_head_stride_bytes;   // bytes between heads for one token
    uint kv_token_stride_bytes;  // bytes between tokens
};

constant uint FLASH_TG_SIZE = 64;
constant uint FLASH_BLOCK_TOKENS = 256;
constant uint FLASH_MAX_HEAD_DIM = 512;
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
        if (simd_lane == 0u) scratch[simd_group] = wave_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            const uint n_groups = (FLASH_TG_SIZE + subgroup_size - 1u) / subgroup_size;
            float merged = -INFINITY;
            for (uint sg = 0u; sg < n_groups; ++sg) merged = fast::max(merged, scratch[sg]);
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
        if (simd_lane == 0u) scratch[simd_group] = wave_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            const uint n_groups = (FLASH_TG_SIZE + subgroup_size - 1u) / subgroup_size;
            float merged = 0.0f;
            for (uint sg = 0u; sg < n_groups; ++sg) merged += scratch[sg];
            scratch[0] = merged;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return scratch[0];
    }
    return wave_sum;
}

inline float4 loadQ8_0Vec4(device const uchar* base, uint vec4_idx) {
    const uint block_idx = vec4_idx >> 3u;          // 8 vec4s per 32-element block
    const uint vec4_in_block = vec4_idx & 7u;
    device const uchar* block = base + block_idx * 34u;
    const float scale = float(as_type<half>(*(device const ushort*)(block)));
    device const packed_char4* quants = (device const packed_char4*)(block + 2u);
    const char4 q = char4(quants[vec4_in_block]);
    return float4(float(q[0]), float(q[1]), float(q[2]), float(q[3])) * scale;
}

kernel void main0(
    constant BatchedFlashAttnQ8Push& p [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const uchar* k_cache [[buffer(2)]],
    device const uchar* v_cache [[buffer(3)]],
    device float* out [[buffer(4)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint subgroup_size [[thread_execution_width]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint head = group_id.x;
    const uint query_idx = group_id.y;
    const uint tid = tid3.x;

    const uint q_per_kv = max(p.n_heads / max(p.n_kv_heads, 1u), 1u);
    const uint kv_head = head / q_per_kv;
    const uint vec4_dim = p.head_dim >> 2;
    const float scale = rsqrt((float)p.head_dim);

    // Causal: query at position (kv_pos_offset + query_idx) attends to
    // KV entries 0..(kv_pos_offset + query_idx) inclusive.
    const uint causal_len = p.kv_pos_offset + query_idx + 1;

    // Q layout matches flash_attn_batched: q[query_idx * n_heads * head_dim + head * head_dim]
    const uint q_base = (query_idx * p.n_heads + head) * p.head_dim;
    const uint out_base = q_base;

    threadgroup float4 q_cache4[FLASH_MAX_HEAD_VEC4];
    threadgroup float4 acc_cache4[FLASH_MAX_HEAD_VEC4];
    threadgroup float scores[FLASH_BLOCK_TOKENS];
    threadgroup float reduce[FLASH_TG_SIZE];
    threadgroup float running_max;
    threadgroup float running_sum;

    for (uint i = tid; i < vec4_dim; i += FLASH_TG_SIZE) {
        q_cache4[i] = *(device const float4*)(q + q_base + (i << 2));
        acc_cache4[i] = float4(0.0f);
    }
    if (tid == 0) {
        running_max = -INFINITY;
        running_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint block_start = 0; block_start < causal_len; block_start += FLASH_BLOCK_TOKENS) {
        const uint block_tokens = min(FLASH_BLOCK_TOKENS, causal_len - block_start);
        const uint block_base_bytes = block_start * p.kv_token_stride_bytes + kv_head * p.kv_head_stride_bytes;
        float local_max = -INFINITY;

        for (uint token_offset = tid; token_offset < block_tokens; token_offset += FLASH_TG_SIZE) {
            const uint kv_base_bytes = block_base_bytes + token_offset * p.kv_token_stride_bytes;

            float score = 0.0f;
            for (uint i = 0; i < vec4_dim; i++) {
                score += dot(q_cache4[i], loadQ8_0Vec4(k_cache + kv_base_bytes, i));
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

        for (uint vi = tid; vi < vec4_dim; vi += FLASH_TG_SIZE) {
            float4 acc = acc_cache4[vi] * rescale;

            uint kv_base_bytes = block_base_bytes;
            for (uint token_offset = 0; token_offset < block_tokens; token_offset++) {
                acc += loadQ8_0Vec4(v_cache + kv_base_bytes, vi) * scores[token_offset];
                kv_base_bytes += p.kv_token_stride_bytes;
            }

            acc_cache4[vi] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            running_sum = running_sum * rescale + block_sum;
            running_max = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    for (uint vi = tid; vi < vec4_dim; vi += FLASH_TG_SIZE) {
        *(device float4*)(out + out_base + (vi << 2)) = acc_cache4[vi] * inv_sum;
    }
}
