#include <metal_stdlib>
using namespace metal;

struct FlashAttnPush {
    uint head_dim;
    uint n_heads;
    uint n_kv_heads;
    uint seq_len;
    uint page_size;
};

constant uint FLASH_TG_SIZE = 64;
constant uint FLASH_BLOCK_TOKENS = 256;
constant uint FLASH_MAX_HEAD_DIM = 256;
constant uint FLASH_MAX_HEAD_VEC4 = FLASH_MAX_HEAD_DIM / 4;

inline uint kvBaseForToken(
    device const uint* page_table,
    constant FlashAttnPush& p,
    uint kv_head,
    uint token_idx
) {
    const uint page_size = max(p.page_size, 1u);
    const uint page = token_idx / page_size;
    const uint page_offset = token_idx % page_size;
    const uint physical_token = page_table[page] * page_size + page_offset;
    return (physical_token * p.n_kv_heads + kv_head) * p.head_dim;
}

kernel void main0(
    constant FlashAttnPush& p [[buffer(0)]],
    device const uint* page_table [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device const float* k_cache [[buffer(3)]],
    device const float* v_cache [[buffer(4)]],
    device float* out [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint q_per_kv = max(p.n_heads / max(p.n_kv_heads, 1u), 1u);
    const uint kv_head = head / q_per_kv;
    const uint q_base = head * p.head_dim;
    const uint vec4_dim = p.head_dim >> 2;
    const float scale = rsqrt((float)p.head_dim);

    threadgroup float4 q_cache4[FLASH_MAX_HEAD_VEC4];
    threadgroup float4 acc_cache4[FLASH_MAX_HEAD_VEC4];
    threadgroup float scores[FLASH_BLOCK_TOKENS];
    threadgroup uint kv_bases[FLASH_BLOCK_TOKENS];
    threadgroup float reduce[FLASH_TG_SIZE];
    threadgroup float running_max;
    threadgroup float running_sum;

    if (tid < vec4_dim) {
        q_cache4[tid] = *(device const float4*)(q + q_base + (tid << 2));
        acc_cache4[tid] = float4(0.0f);
    }
    if (tid == 0) {
        running_max = -INFINITY;
        running_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint block_start = 0; block_start < p.seq_len; block_start += FLASH_BLOCK_TOKENS) {
        const uint block_tokens = min(FLASH_BLOCK_TOKENS, p.seq_len - block_start);
        float local_max = -INFINITY;

        for (uint token_offset = tid; token_offset < block_tokens; token_offset += FLASH_TG_SIZE) {
            const uint token_idx = block_start + token_offset;
            const uint kv_base = kvBaseForToken(page_table, p, kv_head, token_idx);
            kv_bases[token_offset] = kv_base;

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

        reduce[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = FLASH_TG_SIZE >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] = fast::max(reduce[tid], reduce[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float block_max = reduce[0];
        const float next_max = fast::max(running_max, block_max);

        float local_sum = 0.0f;
        for (uint token_offset = tid; token_offset < block_tokens; token_offset += FLASH_TG_SIZE) {
            const float weight = fast::exp(scores[token_offset] - next_max);
            scores[token_offset] = weight;
            local_sum += weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        reduce[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = FLASH_TG_SIZE >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float block_sum = reduce[0];
        const float rescale = running_sum > 0.0f ? fast::exp(running_max - next_max) : 0.0f;

        if (tid < vec4_dim) {
            float4 acc = acc_cache4[tid] * rescale;
            const uint dim_base = tid << 2;

            for (uint token_offset = 0; token_offset < block_tokens; token_offset++) {
                const float weight = scores[token_offset];
                const float4 vv = *(device const float4*)(v_cache + kv_bases[token_offset] + dim_base);
                acc += vv * weight;
            }

            acc_cache4[tid] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
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
