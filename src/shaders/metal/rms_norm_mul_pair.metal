#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint n;
    float eps;
    uint w_a_offset;
    uint w_b_offset;
};

// Two INDEPENDENT RMS norms in one dispatch.
//
// Each threadgroup picks its (input, output, weights, weight_offset) triple by
// group_id (0 or 1) and computes its own normalization independently. There is
// NO shared reduction across threadgroups — the math is identical to running
// rms_norm_mul.metal twice with different buffers, but the host issues only
// one dispatchThreadgroups call (grid={2,1,1}), saving per-dispatch encode
// overhead on the Gemma decode hot path where post_ffw_norm_2 (moe_out_buf)
// and post_ffw_norm_1 (down_buf) currently dispatch back-to-back without an
// intervening barrier.
//
// Distinct from residual_rms_norm_dual.metal, which fuses ONE input with TWO
// weight outputs (sharing the rsqrt). Here both inputs and both outputs are
// independent.
kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* in_a [[buffer(1)]],
    device float* out_a [[buffer(2)]],
    device const float* w_a [[buffer(3)]],
    device const float* in_b [[buffer(4)]],
    device float* out_b [[buffer(5)]],
    device const float* w_b [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint subgroup_size [[thread_execution_width]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shmem[32]; // one slot per simdgroup

    if (simd_lane == 0) {
        shmem[simd_group] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float* input = (group_id == 0u) ? in_a : in_b;
    device float* output = (group_id == 0u) ? out_a : out_b;
    device const float* weights = (group_id == 0u) ? w_a : w_b;
    const uint w_off = (group_id == 0u) ? p.w_a_offset : p.w_b_offset;

    float sum_sq = 0.0f;
    for (uint i = tid; i < p.n; i += tg_size) {
        const float v = input[i];
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);
    if (simd_lane == 0) {
        shmem[simd_group] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    if (simd_group == 0) {
        const uint n_groups = (tg_size + subgroup_size - 1) / subgroup_size;
        total = (simd_lane < n_groups) ? shmem[simd_lane] : 0.0f;
        total = simd_sum(total);
    }

    threadgroup float shared_rms_inv;
    if (tid == 0) {
        shared_rms_inv = rsqrt((total / float(p.n)) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rms_inv = shared_rms_inv;

    for (uint i = tid; i < p.n; i += tg_size) {
        output[i] = weights[w_off + i] * (input[i] * rms_inv);
    }
}
