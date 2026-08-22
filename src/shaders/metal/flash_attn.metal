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

// Cycle 99: halve FLASH_TG_SIZE from 64 to 32 (= 1 simdgroup on Apple7/Apple9).
// For Qwen3-8B head_dim=128 → vec4_dim=32, the V loop `for (vi = tid; vi < 32;
// vi += FLASH_TG_SIZE)` previously launched only threads with tid<32 — half the
// threadgroup (tid 32..63) idled through the entire V accumulator pass. At
// FLASH_TG_SIZE=32 all 32 threads work the V loop. Additionally, since the
// threadgroup is now a single simdgroup (Apple's wave width = 32), both
// `reduceThreadgroupMax` and the inline post-softmax sum collapse to bare
// `simd_max` / `simd_sum`: the `subgroup_size < FLASH_TG_SIZE` branch becomes
// statically false, eliminating the TG-write of `reduce[simd_group]`, the
// threadgroup barrier guarding it, and the cross-simdgroup merge loop. QK
// pass does 2× per-thread work (8 vs 4 tokens at block_tokens=256, TG_SIZE=32)
// but the existing 4-chain FMA split (cycle 96) keeps the chain depth at the
// 4-cycle FMA latency hiding regime, so the doubling is throughput-amortized
// rather than latency-amortized. Net effect: V-loop utilization 50% → 100%,
// scores barrier remains (cross-thread V reads), all other reductions go
// barrier-free. Gemma head_dim=512 (vec4_dim=128) still works correctly —
// each thread does 4 V iters instead of 2, fully utilized either way.
constant uint FLASH_TG_SIZE = 32;
constant uint FLASH_BLOCK_TOKENS = 256;
// Max head_dim supported. Gemma 4 global attention layers use head_dim=512;
// SWA layers use 256. Sized for the larger value so a single shader handles both.
constant uint FLASH_MAX_HEAD_DIM = 512;
constant uint FLASH_MAX_HEAD_VEC4 = FLASH_MAX_HEAD_DIM / 4;

// Simdgroup-redundant-reduction pattern: each simdgroup lane-0 writes its
// wave-reduced partial into scratch, then after a single threadgroup barrier
// every thread re-reads all partials and merges them locally. Saves the
// second broadcast barrier vs the prior "tid==0 merges, barrier, all read"
// idiom. Same pattern that won in cycles 15/16/17 for the rms_norm shaders.
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
        const uint n_groups = (FLASH_TG_SIZE + subgroup_size - 1u) / subgroup_size;
        float merged = -INFINITY;
        for (uint sg = 0u; sg < n_groups; ++sg) {
            merged = fast::max(merged, scratch[sg]);
        }
        return merged;
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
        const uint n_groups = (FLASH_TG_SIZE + subgroup_size - 1u) / subgroup_size;
        float merged = 0.0f;
        for (uint sg = 0u; sg < n_groups; ++sg) {
            merged += scratch[sg];
        }
        return merged;
    }
    return wave_sum;
}

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
    device const float* sinks [[buffer(6)]],  // per-head attention sink values
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
    const float scale = p.attn_scale_bits != 0u ? as_type<float>(p.attn_scale_bits) : rsqrt((float)p.head_dim);
    const bool contiguous_kv = p.page_size == 0u;
    const uint token_stride = contiguous_kv ? (p.kv_token_stride_bytes / uint(sizeof(float))) : (p.n_kv_heads * p.head_dim);
    const uint kv_head_stride = contiguous_kv ? (p.kv_head_stride_bytes / uint(sizeof(float))) : p.head_dim;
    const bool use_sliding_window = p.sliding_window_size > 0u && p.sliding_window_size < p.seq_len;
    const uint sliding_start = use_sliding_window ? (p.seq_len - p.sliding_window_size) : 0u;

    threadgroup float4 q_cache4[FLASH_MAX_HEAD_VEC4];
    threadgroup float scores[FLASH_BLOCK_TOKENS];
    // Cycle 100: `reduce[]` scratch is dead — both reduceThreadgroup{Max,Sum}
    // were inlined to bare `simd_max`/`simd_sum` since TG=32=simd_width on
    // Apple7/Apple9 makes the cross-simdgroup merge branch statically dead.
    // Frees 128B of threadgroup memory, slightly easing occupancy pressure.
    // running_max/running_sum are per-thread registers (every thread maintains
    // the identical running state in lockstep since rescale/block_max/block_sum
    // are produced by full-threadgroup reductions). Saves the broadcast barrier
    // that previously protected tid==0's update to threadgroup-scoped versions.
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // Cycle 94: keep per-thread V accumulators in registers instead of the
    // threadgroup `acc_cache4` buffer. Because the V loop reads/writes
    // `acc_cache4[vi]` with the strided pattern `vi = tid; vi += FLASH_TG_SIZE`,
    // each thread *always* owns the same set of vi values across every
    // block_start iter (and across the sink-scale + output-writeback passes).
    // So the buffer was acting as expensive per-thread state instead of cross-
    // thread shared state. With FLASH_MAX_HEAD_VEC4=128 and FLASH_TG_SIZE=64,
    // each thread owns at most ceil(128/64)=2 vi values → `float4 acc_local[2]`
    // = 8 floats = 32B per thread (fits easily in registers). Removes a 2 KiB
    // threadgroup allocation, the per-block-iter TG read+write of acc_cache4,
    // and the sink-rescale + output-write TG roundtrips. Mirrors cycle 92's
    // "kill dead-after-use threadgroup roundtrips" philosophy applied to the
    // V accumulator, in the same kernel.
    float4 acc_local[(FLASH_MAX_HEAD_VEC4 + FLASH_TG_SIZE - 1u) / FLASH_TG_SIZE];
    for (uint li = 0u; li < (FLASH_MAX_HEAD_VEC4 + FLASH_TG_SIZE - 1u) / FLASH_TG_SIZE; ++li) {
        acc_local[li] = float4(0.0f);
    }

    // Strided loop: vec4_dim may exceed FLASH_TG_SIZE when head_dim > 256
    // (e.g. Gemma 4 global attention layers use head_dim=512 → vec4_dim=128).
    // Cycle 100: TG=32=simd_width (Apple7/Apple9 single simdgroup) → downgrade
    // the post-load barrier from threadgroup_barrier to simdgroup_barrier.
    // The full threadgroup_barrier on Apple GPUs serializes both simdgroups
    // and threadgroup-memory caches; with a single simdgroup the cross-
    // simdgroup wait is dead, only the in-simdgroup memory ordering is
    // needed. ~1152 barriers/token (32 Q-heads × 36 layers × 1 call/head),
    // each replaced with the cheaper variant.
    for (uint i = tid; i < vec4_dim; i += FLASH_TG_SIZE) {
        q_cache4[i] = *(device const float4*)(q + q_base + (i << 2));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Cycle 92: keep per-thread scores in a register array across the
    // QK→softmax pipeline. The two strided loops below visit the same
    // per-thread indices {tid, tid+64, …}; the legacy form wrote each
    // `score` into `scores[token_offset]` (TG memory) at end-of-QK and
    // re-read it as `scores[token_offset]` at top-of-softmax — a pure
    // intra-thread roundtrip through TG memory. With FLASH_BLOCK_TOKENS=256
    // and FLASH_TG_SIZE=64, every thread owns at most 4 tokens per block,
    // so a stack-resident `float local_scores[4]` fits comfortably (16B).
    // The softmax write of `weight` back to `scores[]` is preserved
    // because the V loop reads weights cross-thread (guarded by the
    // existing barrier after softmax). Net: 1 TG-write + 1 TG-read removed
    // per (token,thread) per block per (head×layer). Mirrors the philosophy
    // of cycle 91's "kill dead-after-use threadgroup roundtrips" cleanups.
    float local_scores[FLASH_BLOCK_TOKENS / FLASH_TG_SIZE];

    for (uint block_start = 0u; block_start < p.seq_len; block_start += FLASH_BLOCK_TOKENS) {
        const uint block_tokens = min(FLASH_BLOCK_TOKENS, p.seq_len - block_start);
        // Skip blocks entirely below the sliding window. Besides saving the
        // masked-token loop, this fixes a NaN poisoning bug: a fully-masked
        // block yields block_max = -INF while running_max is still -INF, so
        // the softmax computes exp(-INF - (-INF)) = exp(NaN) — the NaN weight
        // contaminates the V accumulator and the final output becomes NaN
        // (garbage tokens) for every seq_len > sliding_window_size + 256.
        if (use_sliding_window && block_start + block_tokens <= sliding_start) {
            continue;
        }
        const uint block_base = (block_start * token_stride) + kv_head * kv_head_stride;
        float local_max = -INFINITY;

        uint local_idx = 0u;
        for (uint token_offset = tid; token_offset < block_tokens; token_offset += FLASH_TG_SIZE) {
            const uint token_idx = block_start + token_offset;
            if (use_sliding_window && token_idx < sliding_start) {
                local_scores[local_idx++] = -INFINITY;
                continue;
            }
            const uint kv_base = contiguous_kv
                ? (block_base + token_offset * token_stride)
                : kvBaseForToken(page_table, p, kv_head, token_idx);

            // Cycle 96: extend cycle 95's 2-chain QK split to 4 independent
            // chains (score4a/b/c/d) over i%4. With Apple GPU FMA latency
            // ~4 cycles and 1 FMA/cycle issue, 4 chains saturate issue at
            // 4/4 = 1 FMA/cycle (the hardware throughput limit). For
            // vec4_dim=32 (Qwen3-8B head_dim=128), the per-chain depth
            // drops from 16 to 8 FMAs, ~32 cycles latency-bound per chain
            // vs ~64 in cycle 95. For vec4_dim=128 (Gemma head_dim=512),
            // depth drops from 64 to 32. Both vec4_dim values are divisible
            // by 4 so the tail is dead code. Same compiler-hint philosophy
            // as cycle 93's V accumulator 4→8-wide split: more independent
            // chains let the compiler issue FMAs back-to-back instead of
            // stalling on the prior result. Final collapse uses a balanced
            // tree `(a+b)+(c+d)` so the reduction adds only 2 extra ops.
            // flash_attn fires ~1152×/token on Qwen3-8B decode (32 Q-heads
            // × 36 layers); QK reduction dominates the kernel's critical
            // path at short context (block_tokens ≪ FLASH_BLOCK_TOKENS).
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
        // No threadgroup barrier here: scores writes above are at per-thread
        // indices {tid, tid+64, ...} and the softmax loop below reads the
        // same per-thread indices — pure intra-thread dependency. The cross-
        // thread scores read happens in the V loop, which is guarded by the
        // barrier after the softmax loop. reduceThreadgroupMax uses thread-
        // local local_max and has its own internal barrier on the scratch
        // array (separate shmem region from scores).

        // Cycle 100: TG=32=simd_width → block_max collapses to `simd_max`
        // (the helper's subgroup_size<FLASH_TG_SIZE branch is statically
        // false in the actual dispatch shape, but inlining lets the
        // compiler erase the unused `reduce` scratch slot too).
        const float block_max = simd_max(local_max);
        const float next_max = fast::max(running_max, block_max);

        float local_sum = 0.0f;
        local_idx = 0u;
        for (uint token_offset = tid; token_offset < block_tokens; token_offset += FLASH_TG_SIZE) {
            const float weight = fast::exp(local_scores[local_idx++] - next_max);
            scores[token_offset] = weight;
            local_sum += weight;
        }

        // Cycle 98: fuse the post-softmax `scores[]` barrier with the
        // block_sum reduce-write barrier so a single threadgroup_barrier
        // covers both writes. Cycle 100: TG=32=simd_width on Apple7/Apple9
        // means there is exactly one simdgroup per dispatch, so the
        // `subgroup_size < FLASH_TG_SIZE` branch is statically false — the
        // reduce[] write, its scratch slot, and the cross-simdgroup merge
        // are all dead code. `block_sum = simd_sum(local_sum)` is fully
        // wave-resident. The barrier downgrades from threadgroup_barrier
        // to simdgroup_barrier because the only cross-thread reader is
        // the V loop's `scores[token_offset]` reads, all within the same
        // simdgroup.
        const float block_sum = simd_sum(local_sum);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const float rescale = running_sum > 0.0f ? fast::exp(running_max - next_max) : 0.0f;

        // Strided loop over head_dim slices when vec4_dim > FLASH_TG_SIZE.
        // Cycle 93: extend cycle 91's 4-wide V accumulator to 8-wide
        // (acc0..acc7 over 8 staggered tokens). With ~133 V·scores FMAs per
        // (vi, block) for Qwen3-8B decode at n=128, the 4-wide form had 4
        // serial chains of depth ~33 FMAs each; the 8-wide form has 8 chains
        // of depth ~17 — halving the critical-path FMA latency if the loop
        // is latency-bound (which it is at chain-depth ~33 × FMA latency
        // ≫ throughput-limit). 8 float4 accumulators + 8 transient float4
        // V loads = 64 32-bit regs of pressure, well within Apple GPU per-
        // thread register budget. Tail at 4-wide → 1-wide preserves
        // correctness for residual tokens. Same associativity argument as
        // cycles 90/91: sum-of-(v*s) = sum-of-sums by reassociation, within
        // existing fast::math tolerance. Collapse with a balanced 3-level
        // tree `((a0+a1)+(a2+a3)) + ((a4+a5)+(a6+a7))` to keep the final
        // reduction depth at log2(8)=3 instead of serial.
        uint li = 0u;
        for (uint vi = tid; vi < vec4_dim; vi += FLASH_TG_SIZE) {
            float4 acc0 = acc_local[li] * rescale;
            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);
            float4 acc3 = float4(0.0f);
            float4 acc4 = float4(0.0f);
            float4 acc5 = float4(0.0f);
            float4 acc6 = float4(0.0f);
            float4 acc7 = float4(0.0f);
            const uint dim_base = vi << 2;

            if (contiguous_kv) {
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
            } else {
                uint t = 0;
                for (; t + 8u <= block_tokens; t += 8u) {
                    const uint kb0 = kvBaseForToken(page_table, p, kv_head, block_start + t + 0u);
                    const uint kb1 = kvBaseForToken(page_table, p, kv_head, block_start + t + 1u);
                    const uint kb2 = kvBaseForToken(page_table, p, kv_head, block_start + t + 2u);
                    const uint kb3 = kvBaseForToken(page_table, p, kv_head, block_start + t + 3u);
                    const uint kb4 = kvBaseForToken(page_table, p, kv_head, block_start + t + 4u);
                    const uint kb5 = kvBaseForToken(page_table, p, kv_head, block_start + t + 5u);
                    const uint kb6 = kvBaseForToken(page_table, p, kv_head, block_start + t + 6u);
                    const uint kb7 = kvBaseForToken(page_table, p, kv_head, block_start + t + 7u);
                    acc0 += *(device const float4*)(v_cache + kb0 + dim_base) * scores[t + 0u];
                    acc1 += *(device const float4*)(v_cache + kb1 + dim_base) * scores[t + 1u];
                    acc2 += *(device const float4*)(v_cache + kb2 + dim_base) * scores[t + 2u];
                    acc3 += *(device const float4*)(v_cache + kb3 + dim_base) * scores[t + 3u];
                    acc4 += *(device const float4*)(v_cache + kb4 + dim_base) * scores[t + 4u];
                    acc5 += *(device const float4*)(v_cache + kb5 + dim_base) * scores[t + 5u];
                    acc6 += *(device const float4*)(v_cache + kb6 + dim_base) * scores[t + 6u];
                    acc7 += *(device const float4*)(v_cache + kb7 + dim_base) * scores[t + 7u];
                }
                for (; t + 4u <= block_tokens; t += 4u) {
                    const uint kb0 = kvBaseForToken(page_table, p, kv_head, block_start + t + 0u);
                    const uint kb1 = kvBaseForToken(page_table, p, kv_head, block_start + t + 1u);
                    const uint kb2 = kvBaseForToken(page_table, p, kv_head, block_start + t + 2u);
                    const uint kb3 = kvBaseForToken(page_table, p, kv_head, block_start + t + 3u);
                    acc0 += *(device const float4*)(v_cache + kb0 + dim_base) * scores[t + 0u];
                    acc1 += *(device const float4*)(v_cache + kb1 + dim_base) * scores[t + 1u];
                    acc2 += *(device const float4*)(v_cache + kb2 + dim_base) * scores[t + 2u];
                    acc3 += *(device const float4*)(v_cache + kb3 + dim_base) * scores[t + 3u];
                }
                for (; t < block_tokens; t++) {
                    const uint kv_base = kvBaseForToken(page_table, p, kv_head, block_start + t);
                    acc0 += *(device const float4*)(v_cache + kv_base + dim_base) * scores[t];
                }
            }

            acc_local[li] = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7));
            ++li;
        }
        // No threadgroup_barrier here: acc_cache4 is accessed exclusively via
        // per-thread strided indexing (vi = tid; vi += FLASH_TG_SIZE), so each
        // thread reads back only what it wrote — pure intra-thread dependency.
        // The next iteration's first cross-thread threadgroup-memory read is
        // scores[token_offset] in the V loop, which is protected by the
        // softmax barrier. Saves ~1152 barriers/token on Qwen3-8B decode
        // (32 Q-heads × 36 layers × 1 block_start iter at avg ctx ≤256).
        running_sum = running_sum * rescale + block_sum;
        running_max = next_max;
    }

    // Attention sink: per-head learned scalar acts as virtual token in softmax.
    // Absorbs excess attention weight (no V contribution). sinks[head] = NaN means disabled.
    float final_sum = running_sum;
    const float sink_val = sinks[head];
    if (!isnan(sink_val)) {
        const float sink_max = fast::max(running_max, sink_val);
        const float rescale_s = running_sum > 0.0f ? fast::exp(running_max - sink_max) : 0.0f;
        final_sum = running_sum * rescale_s + fast::exp(sink_val - sink_max);
        uint li = 0u;
        for (uint vi = tid; vi < vec4_dim; vi += FLASH_TG_SIZE) {
            acc_local[li++] *= rescale_s;
        }
    }

    // fast::divide maps to Apple GPU hardware reciprocal+mul (vs precise IEEE
    // division), saving ~10 cycles per call. Fires once per Q head per layer
    // (~1152 calls/token on Qwen3-8B: 32 heads × 36 layers). Companion to the
    // fast::exp uses above in the running softmax; precision stays within the
    // already-accepted FA approximation budget.
    const float inv_sum = final_sum > 0.0f ? fast::divide(1.0f, final_sum) : 0.0f;
    uint li_out = 0u;
    for (uint vi = tid; vi < vec4_dim; vi += FLASH_TG_SIZE) {
        *(device float4*)(out + q_base + (vi << 2)) = acc_local[li_out++] * inv_sum;
    }
}
