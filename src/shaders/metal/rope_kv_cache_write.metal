#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Fused single-token RoPE-Q + RoPE-K + KV cache write. Adapted from
// llama.cpp `ggml_metal_op_rope_set_rows` which folds rope output into the
// destination KV slot in one kernel; extended here to also rotate the Q
// vector in the same dispatch so the dense Gemma decode path collapses
// (Q-rope, K-rope+kv-write, V-norm+kv-write) into a single kernel,
// removing one dispatch per dense full-attn layer (≈60/token on Gemma 31B).
//
// Each threadgroup handles one head slot:
//   - if head < n_q_heads: rotates q_inout[head] in place (Q stream).
//   - else (kv_head = head - n_q_heads): rotates K[kv_head] into the cache
//     slot, copies V[kv_head] into the cache (RMS-normalized when
//     apply_v_norm != 0).
//
// When apply_v_norm != 0, V is RMS-normalized per head with unit weights
// before being written to v_cache (Gemma SWA path). This subsumes the
// separate `dispatchRmsNormOnCmd` over v_buf that previously preceded this
// kernel.

struct RopeKvCacheWritePush {
    uint stride;        // elements per head (head_dim)
    uint rope_dim;      // number of rotary dimensions per head (<= stride)
    uint n_q_heads;     // grid slots [0, n_q_heads) handle Q-rope; later slots handle K/V
    uint position;      // token position for this step
    uint dst_offset;    // element offset into kv_k_cache / kv_v_cache (= position * kv_dim)
    uint apply_v_norm;  // 0 = copy V verbatim; nonzero = RMS-normalize V (unit weights)
    float eps;          // RMS norm epsilon (only used when apply_v_norm != 0)
};

kernel void main0(
    constant RopeKvCacheWritePush& p [[buffer(0)]],
    device float* q_inout      [[buffer(1)]],
    device const float* k_in   [[buffer(2)]],
    device const float* v_in   [[buffer(3)]],
    device const float* freqs  [[buffer(4)]],
    device float* k_cache      [[buffer(5)]],
    device float* v_cache      [[buffer(6)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint subgroup_size [[thread_execution_width]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint stride = p.stride;
    const uint half_rot = p.rope_dim / 2;

    if (head < p.n_q_heads) {
        // Q rotary: rotate in place. Pass-through dims (rope_dim..stride) are
        // left untouched, matching the standalone rope_native kernel.
        const uint base = head * stride;
        for (uint i = tid; i < half_rot; i += 64) {
            const float theta = float(p.position) * freqs[i];
            const float cos_t = cos(theta);
            const float sin_t = sin(theta);
            const float x0 = q_inout[base + i];
            const float x1 = q_inout[base + i + half_rot];
            q_inout[base + i] = x0 * cos_t - x1 * sin_t;
            q_inout[base + i + half_rot] = x0 * sin_t + x1 * cos_t;
        }
        return;
    }

    const uint kv_head = head - p.n_q_heads;
    const uint base = kv_head * stride;
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
