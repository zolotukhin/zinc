#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint n;
    float eps;
};

// Triple-fused: residual_norm + residual_add + output_norm.
// Replaces (post_ffn_norm in-place) + barrier + (residual_rms_norm) with one
// dispatch + one barrier on the dense Gemma FFN/next-attn transition.
//
//   res_n[i] = residual_w[i] * residual[i] * rsqrt(mean(residual^2) + eps)
//   hidden[i] += res_n[i]
//   norm_out[i] = output_w[i] * hidden[i] * rsqrt(mean(hidden^2) + eps)
//
// Adapted from residual_rms_norm.metal (existing in-tree fusion) and
// llama.cpp `ggml-metal-ops.cpp::ggml_metal_op_rms_norm` op-fusion idea
// (residual+norm in one pass), extended to two reductions.
//
// 256 threads / 8 simdgroups per threadgroup. One token per threadgroup
// (group_id selects the row); register-cache hidden between the two
// reductions to avoid a third pass over the buffer.
#define N_SIMDGROUPS 8
#define SIMD_WIDTH 32
#define TG_SIZE (N_SIMDGROUPS * SIMD_WIDTH)
#define MAX_PER_THREAD 64

kernel void main0(
    constant Params& p [[buffer(0)]],
    device float* hidden [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device const float* residual_w [[buffer(3)]],
    device float* norm_out [[buffer(4)]],
    device const float* output_w [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial_sums[N_SIMDGROUPS];

    const uint base = group_id * p.n;

    // Pass 1: read residual, accumulate sum of squares for residual norm.
    float sum_sq_r = 0.0f;
    for (uint i = tid; i < p.n; i += TG_SIZE) {
        const float r = residual[base + i];
        sum_sq_r += r * r;
    }

    float sg_sum = simd_sum(sum_sq_r);
    if (lane == 0) partial_sums[sg_idx] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx == 0) {
        float v = (lane < N_SIMDGROUPS) ? partial_sums[lane] : 0.0f;
        float t = simd_sum(v);
        if (lane == 0) partial_sums[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rms_inv_r = rsqrt((partial_sums[0] / float(p.n)) + p.eps);

    // Pass 2: hidden += residual_w[i] * residual[i] * rms_inv_r;
    // accumulate sum of squares for hidden norm; cache new hidden in registers.
    float h_vals[MAX_PER_THREAD];
    float sum_sq_h = 0.0f;
    uint count = 0;
    for (uint i = tid; i < p.n; i += TG_SIZE) {
        const float r = residual[base + i];
        const float r_normed = residual_w[i] * (r * rms_inv_r);
        const float h = hidden[base + i] + r_normed;
        hidden[base + i] = h;
        h_vals[count++] = h;
        sum_sq_h += h * h;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    sg_sum = simd_sum(sum_sq_h);
    if (lane == 0) partial_sums[sg_idx] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx == 0) {
        float v = (lane < N_SIMDGROUPS) ? partial_sums[lane] : 0.0f;
        float t = simd_sum(v);
        if (lane == 0) partial_sums[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rms_inv_h = rsqrt((partial_sums[0] / float(p.n)) + p.eps);

    // Pass 3: norm_out = output_w[i] * h * rms_inv_h.
    count = 0;
    for (uint i = tid; i < p.n; i += TG_SIZE) {
        norm_out[base + i] = output_w[i] * (h_vals[count++] * rms_inv_h);
    }
}
