#include <metal_stdlib>
using namespace metal;

// Batched Rotary Position Embedding for prefill.
// Rotates rope_dim elements per head, for n_tokens tokens at positions
// [position_base, position_base + n_tokens).
//
// Grid: (n_heads, n_tokens, 1), threadgroup: 64 threads.
// Buffer layout: x/y are [n_tokens × n_heads × stride] f32 contiguous.

struct Params {
    uint stride;           // full head dimension (spacing between heads)
    uint rope_dim;         // dimensions to rotate (<= stride, even)
    uint n_heads;
    uint position_base;    // position of the first token in the batch
    uint freq_base_bits;   // RoPE base-frequency bits (0 = use inv_freq buffer)
    uint attn_scale_bits;  // 0 = scale 1.0
};

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const float* inv_freq [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group_id3 [[threadgroup_position_in_grid]]
) {
    const uint tid = tid3.x;
    const uint head = group_id3.x;
    const uint tok = group_id3.y;
    const uint position = p.position_base + tok;

    const uint base_idx = (tok * p.n_heads + head) * p.stride;
    const uint half_rot = p.rope_dim / 2;

    const float freq_base = as_type<float>(p.freq_base_bits);
    const float attn_scale = p.attn_scale_bits != 0u ? as_type<float>(p.attn_scale_bits) : 1.0f;
    const bool use_freq_buf = (p.freq_base_bits == 0u);

    for (uint i = tid; i < half_rot; i += 64) {
        const float x_i = x[base_idx + i];
        const float x_ihd = x[base_idx + i + half_rot];

        float freq_i;
        if (use_freq_buf) {
            freq_i = inv_freq[i];
        } else {
            const float exponent = float(2u * i) / float(p.rope_dim);
            freq_i = 1.0f / pow(freq_base, exponent);
        }
        const float theta = float(position) * freq_i;
        const float cos_t = cos(theta) * attn_scale;
        const float sin_t = sin(theta) * attn_scale;

        y[base_idx + i]            = x_i * cos_t - x_ihd * sin_t;
        y[base_idx + i + half_rot] = x_i * sin_t + x_ihd * cos_t;
    }

    for (uint i = p.rope_dim + tid; i < p.stride; i += 64) {
        y[base_idx + i] = x[base_idx + i];
    }
}
