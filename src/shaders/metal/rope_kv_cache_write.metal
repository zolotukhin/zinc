#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Fused single-token RoPE-K + KV cache write. Adapted from
// llama.cpp `ggml_metal_op_rope_set_rows` which folds rope output into the
// destination KV slot in one kernel, removing the rope→kv-write barrier.
//
// Each threadgroup handles one KV head:
//   - applies RoPE to the first rope_dim elements of k_in[head] and writes
//     the rotated values directly into k_cache[dst_offset + head*stride ..]
//   - copies pass-through K elements (rope_dim..stride) verbatim
//   - copies v_in[head] into v_cache[dst_offset + head*stride ..]
//
// When apply_v_norm != 0, V is RMS-normalized per head with unit weights
// before being written to v_cache (Gemma SWA path). This subsumes the
// separate `dispatchRmsNormOnCmd` over v_buf that previously preceded this
// kernel, saving one dispatch per dense full-attn layer.

struct RopeKvCacheWritePush {
    uint stride;        // elements per head (head_dim)
    uint rope_dim;      // number of rotary dimensions per head (<= stride)
    uint position;      // token position for this step
    uint dst_offset;    // element offset into kv_k_cache / kv_v_cache (= position * kv_dim)
    uint apply_v_norm;  // 0 = copy V verbatim; nonzero = RMS-normalize V (unit weights)
    float eps;          // RMS norm epsilon (only used when apply_v_norm != 0)
};

kernel void main0(
    constant RopeKvCacheWritePush& p [[buffer(0)]],
    device const float* k_in   [[buffer(1)]],
    device const float* v_in   [[buffer(2)]],
    device const float* freqs  [[buffer(3)]],
    device float* k_cache      [[buffer(4)]],
    device float* v_cache      [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint subgroup_size [[thread_execution_width]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint stride = p.stride;
    const uint half_rot = p.rope_dim / 2;
    const uint base = head * stride;
    const uint dst_base = p.dst_offset + base;

    // K rotary: apply RoPE pair (i, i+half_rot) and write to cache.
    for (uint i = tid; i < half_rot; i += 64) {
        const float theta = float(p.position) * freqs[i];
        const float cos_t = cos(theta);
        const float sin_t = sin(theta);
        const float x0 = k_in[base + i];
        const float x1 = k_in[base + i + half_rot];
        k_cache[dst_base + i] = x0 * cos_t - x1 * sin_t;
        k_cache[dst_base + i + half_rot] = x0 * sin_t + x1 * cos_t;
    }

    // K pass-through: dimensions beyond rope_dim are copied verbatim.
    for (uint i = p.rope_dim + tid; i < stride; i += 64) {
        k_cache[dst_base + i] = k_in[base + i];
    }

    if (p.apply_v_norm != 0u) {
        // Compute per-head sum of squares for V, threadgroup-reduce, then
        // multiply each element by rsqrt(mean + eps) while writing to v_cache.
        // Mirrors `rms_norm_mul.metal` but folds the read/normalize/write into
        // the same kernel that performs the V cache copy.
        threadgroup float shmem[2]; // 64 threads / 32 lanes = 2 simdgroups

        float sum_sq = 0.0f;
        for (uint i = tid; i < stride; i += 64) {
            const float v = v_in[base + i];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        if (simd_lane == 0) {
            shmem[simd_group] = sum_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float shared_rms_inv;
        if (tid == 0) {
            const float total = shmem[0] + shmem[1];
            shared_rms_inv = rsqrt((total / float(stride)) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float rms_inv = shared_rms_inv;

        for (uint i = tid; i < stride; i += 64) {
            v_cache[dst_base + i] = v_in[base + i] * rms_inv;
        }
    } else {
        // V copy.
        for (uint i = tid; i < stride; i += 64) {
            v_cache[dst_base + i] = v_in[base + i];
        }
    }
}
