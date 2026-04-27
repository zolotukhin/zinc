#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint n;
    float eps;
    uint weight_offset;
};

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint subgroup_size [[thread_execution_width]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shmem[32];

    if (simd_lane == 0) {
        shmem[simd_group] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint base = group_id * p.n;

    float sum_sq = 0.0f;
    for (uint i = tid; i < p.n; i += tg_size) {
        const float v = input[base + i];
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
        output[base + i] = weights[p.weight_offset + i] * (input[base + i] * rms_inv);
    }
}
