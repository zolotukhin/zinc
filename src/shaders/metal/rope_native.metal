#include <metal_stdlib>
using namespace metal;

// Native RoPE kernel that reads precomputed inverse frequencies from a buffer.
// Supports YaRN and other frequency-scaling schemes (the host precomputes the
// scaled frequencies; the kernel just reads them).

struct RopeNativePush {
    uint stride;    // elements per head (head_dim)
    uint rope_dim;  // number of rotary dimensions (<= stride)
    uint n_heads;   // number of heads
    uint position;  // token position for this step
};

kernel void main0(
    constant RopeNativePush& p [[buffer(0)]],
    device float* data        [[buffer(1)]],
    device const float* freqs [[buffer(2)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    const uint half_rot = p.rope_dim / 2;
    const uint base = head * p.stride;

    // Apply rotary embedding to the first rope_dim elements
    for (uint i = tid; i < half_rot; i += 64) {
        const float theta = float(p.position) * freqs[i];
        const float cos_t = cos(theta);
        const float sin_t = sin(theta);

        const uint idx0 = base + i;
        const uint idx1 = base + i + half_rot;
        const float x0 = data[idx0];
        const float x1 = data[idx1];

        data[idx0] = x0 * cos_t - x1 * sin_t;
        data[idx1] = x0 * sin_t + x1 * cos_t;
    }

    // Pass through non-rotary dimensions (rope_dim..stride) unchanged
}
