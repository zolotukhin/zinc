#include <metal_stdlib>
using namespace metal;

// Batched causal flash attention for prefill.
// Processes N_queries queries simultaneously, each with causal masking
// (query at position t attends to KV positions 0..t only).
//
// Grid: (n_heads, n_queries, 1)
//   - grid.x = head index
//   - grid.y = query index (0..n_queries-1)
//
// Q buffer layout: [n_queries × n_heads × head_dim] contiguous
//   q[query * n_heads * head_dim + head * head_dim + dim]
//
// KV cache layout: [max_seq × n_kv_heads × head_dim] contiguous
//   k_cache[token * n_kv_heads * head_dim + kv_head * head_dim + dim]
//
// Output layout: [n_queries × n_heads × head_dim] matching Q layout

struct BatchedFlashAttnPush {
    uint head_dim;
    uint n_heads;
    uint n_kv_heads;
    uint kv_len;        // total KV entries (= kv_pos_offset + n_queries for contiguous prefill)
    uint n_queries;     // number of query tokens
    uint kv_pos_offset; // position offset for the first query in the KV cache
};

constant uint FLASH_TG_SIZE = 64;
constant uint FLASH_BLOCK_TOKENS = 256;
// Max head_dim supported. Gemma 4 global attention layers use head_dim=512.
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

kernel void main0(
    constant BatchedFlashAttnPush& p [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* k_cache [[buffer(2)]],
    device const float* v_cache [[buffer(3)]],
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
    const uint token_stride = p.n_kv_heads * p.head_dim;

    // Causal masking: this query at position (kv_pos_offset + query_idx) can attend
    // to KV entries 0..(kv_pos_offset + query_idx) inclusive.
    const uint causal_len = p.kv_pos_offset + query_idx + 1;

    // Q base: q[query_idx * n_heads * head_dim + head * head_dim]
    const uint q_base = (query_idx * p.n_heads + head) * p.head_dim;

    // Output base: same layout as Q
    const uint out_base = q_base;

    threadgroup float4 q_cache4[FLASH_MAX_HEAD_VEC4];
    threadgroup float4 acc_cache4[FLASH_MAX_HEAD_VEC4];
    threadgroup float scores[FLASH_BLOCK_TOKENS];
    threadgroup float reduce[FLASH_TG_SIZE];
    threadgroup float running_max;
    threadgroup float running_sum;

    // Strided loop in case vec4_dim > FLASH_TG_SIZE (head_dim=512 → vec4_dim=128).
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
        const uint block_base = (block_start * token_stride) + kv_head * p.head_dim;
        float local_max = -INFINITY;

        for (uint token_offset = tid; token_offset < block_tokens; token_offset += FLASH_TG_SIZE) {
            const uint kv_base = block_base + token_offset * token_stride;

            float score = 0.0f;
            for (uint i = 0; i < vec4_dim; i++) {
                const float4 qv = q_cache4[i];
                const float4 kv = *(device const float4*)(k_cache + kv_base + (i << 2));
                score += dot(qv, kv);
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
            const uint dim_base = vi << 2;

            uint kv_base = block_base + dim_base;
            for (uint token_offset = 0; token_offset < block_tokens; token_offset++) {
                acc += *(device const float4*)(v_cache + kv_base) * scores[token_offset];
                kv_base += token_stride;
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
