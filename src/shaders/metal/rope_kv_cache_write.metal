#include <metal_stdlib>
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
// V is not rotated; the host applies any V-side RMS norm before this dispatch.

struct RopeKvCacheWritePush {
    uint stride;     // elements per head (head_dim)
    uint rope_dim;   // number of rotary dimensions per head (<= stride)
    uint position;   // token position for this step
    uint dst_offset; // element offset into kv_k_cache / kv_v_cache (= position * kv_dim)
};

kernel void main0(
    constant RopeKvCacheWritePush& p [[buffer(0)]],
    device const float* k_in   [[buffer(1)]],
    device const float* v_in   [[buffer(2)]],
    device const float* freqs  [[buffer(3)]],
    device float* k_cache      [[buffer(4)]],
    device float* v_cache      [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
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

    // V copy.
    for (uint i = tid; i < stride; i += 64) {
        v_cache[dst_base + i] = v_in[base + i];
    }
}
