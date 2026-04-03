#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint n;
    float eps;
    float scale;
};

// Fused residual-add + RMS norm: hidden += scale * residual; norm = weights * normalize(hidden)
// Replaces scale_accumulate → barrier → rms_norm_mul, eliminating one barrier per layer.
//
// Uses 256 threads (8 simdgroups) per threadgroup for fast elementwise throughput
// with threadgroup-memory reduction for the sum-of-squares.
// Pass 1 stores intermediate values in registers to avoid re-reading hidden in pass 2.
#define N_SIMDGROUPS 8
#define SIMD_WIDTH 32
#define TG_SIZE (N_SIMDGROUPS * SIMD_WIDTH)
#define MAX_PER_THREAD 128

kernel void main0(
    constant Params& p [[buffer(0)]],
    device float* hidden [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* norm_out [[buffer(3)]],
    device const float* weights [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial_sums[N_SIMDGROUPS];

    const uint base = group_id * p.n;
    const float scale = p.scale;

    // Pass 1: residual add + accumulate sum of squares.
    // Store residual-added values in register array to avoid re-reading hidden in pass 2.
    float vals[MAX_PER_THREAD];
    float sum_sq = 0.0f;
    uint count = 0;
    for (uint i = tid; i < p.n; i += TG_SIZE) {
        const float h = fma(scale, residual[base + i], hidden[base + i]);
        hidden[base + i] = h;
        vals[count++] = h;
        sum_sq += h * h;
    }

    // Hierarchical reduction: simd_sum per simdgroup, then combine in threadgroup memory.
    float sg_sum = simd_sum(sum_sq);
    if (lane == 0) partial_sums[sg_idx] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup reduces the partial sums.
    float total_sq;
    if (sg_idx == 0) {
        float v = (lane < N_SIMDGROUPS) ? partial_sums[lane] : 0.0f;
        total_sq = simd_sum(v);
        if (lane == 0) partial_sums[0] = total_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float rms_inv = rsqrt((partial_sums[0] / float(p.n)) + p.eps);

    // Pass 2: normalize from registers (avoids re-reading hidden_buf).
    count = 0;
    for (uint i = tid; i < p.n; i += TG_SIZE) {
        norm_out[base + i] = weights[i] * (vals[count++] * rms_inv);
    }
}
