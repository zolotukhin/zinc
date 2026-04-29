#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint n;
    float eps;
    float scale;
    uint weights_b_offset;
};

// Fused residual-add + dual RMS norm:
//   hidden[i] += scale * residual[i]
//   norm_a[i] = weights_a[i] * normalize(hidden)[i]
//   norm_b[i] = weights_b[weights_b_offset + i] * normalize(hidden)[i]
//
// Single threadgroup; reduction in threadgroup memory; values cached in
// registers between Pass 1 and Pass 2 to avoid re-reading hidden_buf.
//
// Use case (Gemma decode): replaces residual_rms_norm + barrier + rms_norm
// (gate_scale) when both ffn_norm and gate_scale outputs derive from the
// same post-residual hidden state. Eliminates one dispatch and one barrier
// per Gemma MoE layer.
#define N_SIMDGROUPS 8
#define SIMD_WIDTH 32
#define TG_SIZE (N_SIMDGROUPS * SIMD_WIDTH)
#define MAX_PER_THREAD 128

kernel void main0(
    constant Params& p [[buffer(0)]],
    device float* hidden [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* norm_out_a [[buffer(3)]],
    device const float* weights_a [[buffer(4)]],
    device float* norm_out_b [[buffer(5)]],
    device const float* weights_b [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial_sums[N_SIMDGROUPS];

    const uint base = group_id * p.n;
    const float scale = p.scale;

    // Pass 1: residual add + accumulate sum of squares.
    float vals[MAX_PER_THREAD];
    float sum_sq = 0.0f;
    uint count = 0;
    for (uint i = tid; i < p.n; i += TG_SIZE) {
        const float h = fma(scale, residual[base + i], hidden[base + i]);
        hidden[base + i] = h;
        vals[count++] = h;
        sum_sq += h * h;
    }

    float sg_sum = simd_sum(sum_sq);
    if (lane == 0) partial_sums[sg_idx] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0) {
        float v = (lane < N_SIMDGROUPS) ? partial_sums[lane] : 0.0f;
        const float total_sq = simd_sum(v);
        if (lane == 0) partial_sums[0] = total_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float rms_inv = rsqrt((partial_sums[0] / float(p.n)) + p.eps);

    // Pass 2: write to BOTH outputs, sharing the rms_inv reduction.
    count = 0;
    const uint b_off = p.weights_b_offset;
    for (uint i = tid; i < p.n; i += TG_SIZE) {
        const float scaled = vals[count++] * rms_inv;
        norm_out_a[base + i] = weights_a[i] * scaled;
        norm_out_b[base + i] = weights_b[b_off + i] * scaled;
    }
}
