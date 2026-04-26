#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint n_experts;
    uint k;
    uint logits_stride;
    uint output_stride;
};

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* logits_data [[buffer(1)]],
    device uint* output_data [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint subgroup_size [[thread_execution_width]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (p.n_experts == 0u || p.n_experts > 256u || p.k == 0u || p.k > p.n_experts) {
        return;
    }

    device const float* logits = logits_data + token_idx * p.logits_stride;
    device uint* output = output_data + token_idx * p.output_stride;

    threadgroup float probs[256];
    threadgroup float reduce_val[64];
    threadgroup float max_val;
    threadgroup float sum_val;
    threadgroup float local_val[64];
    threadgroup uint local_idx[64];

    float local_max = -INFINITY;
    for (uint i = tid; i < p.n_experts; i += 64u) {
        const float v = logits[i];
        probs[i] = v;
        local_max = fast::max(local_max, v);
    }

    const float wave_max = simd_max(local_max);
    if (subgroup_size < 64u) {
        if (simd_lane == 0u) {
            reduce_val[simd_group] = wave_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            const uint n_groups = (64u + subgroup_size - 1u) / subgroup_size;
            float merged = -INFINITY;
            for (uint sg = 0u; sg < n_groups; sg++) {
                merged = fast::max(merged, reduce_val[sg]);
            }
            max_val = merged;
        }
    } else if (tid == 0u) {
        max_val = wave_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint i = tid; i < p.n_experts; i += 64u) {
        const float e = exp(probs[i] - max_val);
        probs[i] = e;
        local_sum += e;
    }

    const float wave_sum = simd_sum(local_sum);
    if (subgroup_size < 64u) {
        if (simd_lane == 0u) {
            reduce_val[simd_group] = wave_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            const uint n_groups = (64u + subgroup_size - 1u) / subgroup_size;
            float merged = 0.0f;
            for (uint sg = 0u; sg < n_groups; sg++) {
                merged += reduce_val[sg];
            }
            sum_val = merged;
        }
    } else if (tid == 0u) {
        sum_val = wave_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_sum = (sum_val > 0.0f) ? (1.0f / sum_val) : 0.0f;
    for (uint i = tid; i < p.n_experts; i += 64u) {
        probs[i] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ki = 0u; ki < p.k; ki++) {
        float best_val = -1.0f;
        uint best_idx = 0u;
        for (uint i = tid; i < p.n_experts; i += 64u) {
            if (probs[i] > best_val) {
                best_val = probs[i];
                best_idx = i;
            }
        }

        local_val[tid] = best_val;
        local_idx[tid] = best_idx;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float global_best = -1.0f;
            uint global_idx = 0u;
            for (uint lane = 0u; lane < 64u; lane++) {
                if (local_val[lane] > global_best) {
                    global_best = local_val[lane];
                    global_idx = local_idx[lane];
                }
            }
            output[ki] = global_idx;
            output[p.k + ki] = as_type<uint>(global_best);
            probs[global_idx] = -1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float wsum = 0.0f;
        for (uint i = 0u; i < p.k; i++) {
            wsum += as_type<float>(output[p.k + i]);
        }
        const float inv_wsum = (wsum > 0.0f) ? (1.0f / wsum) : 0.0f;
        for (uint i = 0u; i < p.k; i++) {
            const float w = as_type<float>(output[p.k + i]) * inv_wsum;
            output[p.k + i] = as_type<uint>(w);
        }
    }
}
