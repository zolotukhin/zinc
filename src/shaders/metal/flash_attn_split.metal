#include <metal_stdlib>
using namespace metal;

struct FlashAttnPush {
    uint head_dim;
    uint n_heads;
    uint n_kv_heads;
    uint seq_len;
    uint sliding_window_size;
    uint page_size;
    uint attn_scale_bits;
    uint kv_head_stride_bytes;
    uint kv_token_stride_bytes;
};

// Split-simdgroup decode flash attention (f32 KV, contiguous layout only).
//
// The single-query decode flash in flash_attn.metal launches ONE 32-thread
// threadgroup per Q head — 32 tg × 32 threads ≈ 1024 threads on a 40-core
// GPU. At long context the kernel is LATENCY-bound: measured 287 us/dispatch
// at ~1000 ctx while touching only ~2 MB of KV (a 71 MB dmmv takes 269 us).
// There is simply no other work resident to hide the L2 latency of the
// KV stream, so decode throughput decays linearly with context while
// llama.cpp (whose flash_attn_ext_vec splits the KV scan across many
// simdgroups) stays flat.
//
// This kernel keeps one threadgroup per Q head but runs FLASH_SPLIT_NSG
// simdgroups. Each simdgroup executes the SAME tuned inner loop as
// flash_attn.metal (4-chain QK FMA split, 8-wide V accumulate, per-block
// online softmax) over an interleaved subset of 64-token key blocks,
// maintaining its own (running_max, running_sum, acc) partial. A final
// in-kernel merge rescales the per-simdgroup partials to the global max —
// the standard split-K flash-attention reduction (mathematically identical
// to the serial online softmax; FP reduction order differs, so this path is
// gated to long context and validated by greedy-match rather than
// byte-compare).
//
// Contiguous KV only: the decode dispatch site always sets page_size=0.
// Sinks are supported (merge-phase epilogue, same math as flash_attn.metal).
// head_dim <= FLASH_SPLIT_MAX_HEAD_DIM (dispatch-gated).

#ifndef ZINC_FLASH_SPLIT_NSG
#define ZINC_FLASH_SPLIT_NSG 8
#endif

constant uint FLASH_SPLIT_NSG = ZINC_FLASH_SPLIT_NSG;
constant uint SIMD_W = 32;
// 64-token blocks (vs the serial kernel's 256) so moderate contexts spread
// across all simdgroups: ctx 500 → 8 blocks → all 8 simdgroups busy.
constant uint FLASH_SPLIT_BLOCK_TOKENS = 64;
constant uint FLASH_SPLIT_MAX_HEAD_DIM = 256;
constant uint FLASH_SPLIT_MAX_HEAD_VEC4 = FLASH_SPLIT_MAX_HEAD_DIM / 4;

kernel void main0(
    constant FlashAttnPush& p [[buffer(0)]],
    device const uint* page_table [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device const float* k_cache [[buffer(3)]],
    device const float* v_cache [[buffer(4)]],
    device float* out [[buffer(5)]],
    device const float* sinks [[buffer(6)]],  // per-head attention sink values
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]]
) {
    const uint q_per_kv = max(p.n_heads / max(p.n_kv_heads, 1u), 1u);
    const uint kv_head = head / q_per_kv;
    const uint q_base = head * p.head_dim;
    const uint vec4_dim = p.head_dim >> 2;
    const float scale = p.attn_scale_bits != 0u ? as_type<float>(p.attn_scale_bits) : rsqrt((float)p.head_dim);
    const uint token_stride = p.kv_token_stride_bytes / uint(sizeof(float));
    const uint kv_head_stride = p.kv_head_stride_bytes / uint(sizeof(float));
    const bool use_sliding_window = p.sliding_window_size > 0u && p.sliding_window_size < p.seq_len;
    const uint sliding_start = use_sliding_window ? (p.seq_len - p.sliding_window_size) : 0u;

    threadgroup float4 q_cache4[FLASH_SPLIT_MAX_HEAD_VEC4];
    // Per-simdgroup score staging (sg-private slices — only simdgroup_barrier
    // needed inside the scan loop, exactly like the serial kernel at TG=32).
    threadgroup float scores_all[FLASH_SPLIT_NSG * FLASH_SPLIT_BLOCK_TOKENS];
    // Merge-phase scratch: per-simdgroup partial accumulators + (max, sum).
    threadgroup float4 acc_merge[FLASH_SPLIT_NSG * FLASH_SPLIT_MAX_HEAD_VEC4];
    threadgroup float part_max[FLASH_SPLIT_NSG];
    threadgroup float part_sum[FLASH_SPLIT_NSG];

    threadgroup float* scores = scores_all + sg_idx * FLASH_SPLIT_BLOCK_TOKENS;

    // Cooperative Q load by the full threadgroup, then one full barrier so
    // every simdgroup sees it.
    for (uint i = tid; i < vec4_dim; i += FLASH_SPLIT_NSG * SIMD_W) {
        q_cache4[i] = *(device const float4*)(q + q_base + (i << 2));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    float4 acc_local[(FLASH_SPLIT_MAX_HEAD_VEC4 + SIMD_W - 1u) / SIMD_W];
    for (uint li = 0u; li < (FLASH_SPLIT_MAX_HEAD_VEC4 + SIMD_W - 1u) / SIMD_W; ++li) {
        acc_local[li] = float4(0.0f);
    }

    float local_scores[FLASH_SPLIT_BLOCK_TOKENS / SIMD_W];

    // Interleaved block assignment: simdgroup sg takes blocks sg, sg+NSG, ...
    for (uint block_start = sg_idx * FLASH_SPLIT_BLOCK_TOKENS;
         block_start < p.seq_len;
         block_start += FLASH_SPLIT_NSG * FLASH_SPLIT_BLOCK_TOKENS)
    {
        const uint block_tokens = min(FLASH_SPLIT_BLOCK_TOKENS, p.seq_len - block_start);
        // Whole block below the sliding window → every score would be -INF;
        // skip the memory traffic entirely.
        if (use_sliding_window && block_start + block_tokens <= sliding_start) {
            continue;
        }
        const uint block_base = (block_start * token_stride) + kv_head * kv_head_stride;
        float local_max = -INFINITY;

        uint local_idx = 0u;
        for (uint token_offset = simd_lane; token_offset < block_tokens; token_offset += SIMD_W) {
            const uint token_idx = block_start + token_offset;
            if (use_sliding_window && token_idx < sliding_start) {
                local_scores[local_idx++] = -INFINITY;
                continue;
            }
            const uint kv_base = block_base + token_offset * token_stride;

            // 4-chain QK FMA split (see flash_attn.metal cycle 96).
            float4 score4a = float4(0.0f);
            float4 score4b = float4(0.0f);
            float4 score4c = float4(0.0f);
            float4 score4d = float4(0.0f);
            uint i = 0u;
            for (; i + 4u <= vec4_dim; i += 4u) {
                const float4 qv0 = q_cache4[i];
                const float4 kv0 = *(device const float4*)(k_cache + kv_base + (i << 2));
                const float4 qv1 = q_cache4[i + 1u];
                const float4 kv1 = *(device const float4*)(k_cache + kv_base + ((i + 1u) << 2));
                const float4 qv2 = q_cache4[i + 2u];
                const float4 kv2 = *(device const float4*)(k_cache + kv_base + ((i + 2u) << 2));
                const float4 qv3 = q_cache4[i + 3u];
                const float4 kv3 = *(device const float4*)(k_cache + kv_base + ((i + 3u) << 2));
                score4a = fma(qv0, kv0, score4a);
                score4b = fma(qv1, kv1, score4b);
                score4c = fma(qv2, kv2, score4c);
                score4d = fma(qv3, kv3, score4d);
            }
            for (; i + 2u <= vec4_dim; i += 2u) {
                const float4 qv0 = q_cache4[i];
                const float4 kv0 = *(device const float4*)(k_cache + kv_base + (i << 2));
                const float4 qv1 = q_cache4[i + 1u];
                const float4 kv1 = *(device const float4*)(k_cache + kv_base + ((i + 1u) << 2));
                score4a = fma(qv0, kv0, score4a);
                score4b = fma(qv1, kv1, score4b);
            }
            for (; i < vec4_dim; ++i) {
                const float4 qv = q_cache4[i];
                const float4 kv = *(device const float4*)(k_cache + kv_base + (i << 2));
                score4a = fma(qv, kv, score4a);
            }
            float score = dot((score4a + score4b) + (score4c + score4d), float4(1.0f));
            score *= scale;
            local_scores[local_idx++] = score;
            local_max = fast::max(local_max, score);
        }

        const float block_max = simd_max(local_max);
        const float next_max = fast::max(running_max, block_max);

        float local_sum = 0.0f;
        local_idx = 0u;
        for (uint token_offset = simd_lane; token_offset < block_tokens; token_offset += SIMD_W) {
            const float weight = fast::exp(local_scores[local_idx++] - next_max);
            scores[token_offset] = weight;
            local_sum += weight;
        }

        const float block_sum = simd_sum(local_sum);
        // scores slice is sg-private; only in-simdgroup ordering needed.
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const float rescale = running_sum > 0.0f ? fast::exp(running_max - next_max) : 0.0f;

        // 8-wide V accumulate (see flash_attn.metal cycle 93), contiguous KV.
        uint li = 0u;
        for (uint vi = simd_lane; vi < vec4_dim; vi += SIMD_W) {
            float4 acc0 = acc_local[li] * rescale;
            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);
            float4 acc3 = float4(0.0f);
            float4 acc4 = float4(0.0f);
            float4 acc5 = float4(0.0f);
            float4 acc6 = float4(0.0f);
            float4 acc7 = float4(0.0f);
            const uint dim_base = vi << 2;

            uint kv_base = block_base + dim_base;
            const uint stride2 = token_stride << 1;
            const uint stride3 = stride2 + token_stride;
            const uint stride4 = token_stride << 2;
            const uint stride5 = stride4 + token_stride;
            const uint stride6 = stride4 + stride2;
            const uint stride7 = stride4 + stride3;
            const uint stride8 = token_stride << 3;
            uint t = 0;
            for (; t + 8u <= block_tokens; t += 8u) {
                const float4 v0 = *(device const float4*)(v_cache + kv_base);
                const float4 v1 = *(device const float4*)(v_cache + kv_base + token_stride);
                const float4 v2 = *(device const float4*)(v_cache + kv_base + stride2);
                const float4 v3 = *(device const float4*)(v_cache + kv_base + stride3);
                const float4 v4 = *(device const float4*)(v_cache + kv_base + stride4);
                const float4 v5 = *(device const float4*)(v_cache + kv_base + stride5);
                const float4 v6 = *(device const float4*)(v_cache + kv_base + stride6);
                const float4 v7 = *(device const float4*)(v_cache + kv_base + stride7);
                acc0 += v0 * scores[t + 0u];
                acc1 += v1 * scores[t + 1u];
                acc2 += v2 * scores[t + 2u];
                acc3 += v3 * scores[t + 3u];
                acc4 += v4 * scores[t + 4u];
                acc5 += v5 * scores[t + 5u];
                acc6 += v6 * scores[t + 6u];
                acc7 += v7 * scores[t + 7u];
                kv_base += stride8;
            }
            for (; t + 4u <= block_tokens; t += 4u) {
                const float4 v0 = *(device const float4*)(v_cache + kv_base);
                const float4 v1 = *(device const float4*)(v_cache + kv_base + token_stride);
                const float4 v2 = *(device const float4*)(v_cache + kv_base + stride2);
                const float4 v3 = *(device const float4*)(v_cache + kv_base + stride3);
                acc0 += v0 * scores[t + 0u];
                acc1 += v1 * scores[t + 1u];
                acc2 += v2 * scores[t + 2u];
                acc3 += v3 * scores[t + 3u];
                kv_base += stride4;
            }
            for (; t < block_tokens; t++) {
                acc0 += *(device const float4*)(v_cache + kv_base) * scores[t];
                kv_base += token_stride;
            }

            acc_local[li] = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7));
            ++li;
        }
        running_sum = running_sum * rescale + block_sum;
        running_max = next_max;
    }

    // ---- Split-K merge: rescale per-simdgroup partials to the global max ----
    {
        uint li = 0u;
        for (uint vi = simd_lane; vi < vec4_dim; vi += SIMD_W) {
            acc_merge[sg_idx * FLASH_SPLIT_MAX_HEAD_VEC4 + vi] = acc_local[li++];
        }
        if (simd_lane == 0u) {
            part_max[sg_idx] = running_max;
            part_sum[sg_idx] = running_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every thread recomputes the (tiny) merge factors in lockstep.
    float global_max = -INFINITY;
    for (uint s = 0u; s < FLASH_SPLIT_NSG; ++s) {
        global_max = fast::max(global_max, part_max[s]);
    }
    float merged_sum = 0.0f;
    float w[FLASH_SPLIT_NSG];
    for (uint s = 0u; s < FLASH_SPLIT_NSG; ++s) {
        const float ws = part_sum[s] > 0.0f ? fast::exp(part_max[s] - global_max) : 0.0f;
        w[s] = ws;
        merged_sum += part_sum[s] * ws;
    }

    // Attention sink (same math as flash_attn.metal, applied on the merged
    // partials): virtual token in the softmax, no V contribution.
    float final_sum = merged_sum;
    float sink_rescale = 1.0f;
    const float sink_val = sinks[head];
    if (!isnan(sink_val)) {
        const float sink_max = fast::max(global_max, sink_val);
        sink_rescale = merged_sum > 0.0f ? fast::exp(global_max - sink_max) : 0.0f;
        final_sum = merged_sum * sink_rescale + fast::exp(sink_val - sink_max);
    }

    const float out_scale = final_sum > 0.0f ? fast::divide(sink_rescale, final_sum) : 0.0f;

    // Full-threadgroup strided output: sum the NSG rescaled partials per slice.
    for (uint vi = tid; vi < vec4_dim; vi += FLASH_SPLIT_NSG * SIMD_W) {
        float4 merged = float4(0.0f);
        for (uint s = 0u; s < FLASH_SPLIT_NSG; ++s) {
            merged = fma(acc_merge[s * FLASH_SPLIT_MAX_HEAD_VEC4 + vi], w[s], merged);
        }
        *(device float4*)(out + q_base + (vi << 2)) = merged * out_scale;
    }
}
