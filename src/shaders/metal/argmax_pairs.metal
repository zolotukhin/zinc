#include <metal_stdlib>
using namespace metal;

struct ArgmaxPairsPush {
    uint n_pairs;
};

kernel void main0(
    device const uint* partials [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant ArgmaxPairsPush& p [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float best_vals[256];
    threadgroup uint best_idxs[256];

    float best_val = -INFINITY;
    uint best_idx = 0u;

    for (uint i = tid; i < p.n_pairs; i += 256u) {
        const uint idx = partials[i * 2u + 0u];
        const float v = as_type<float>(partials[i * 2u + 1u]);
        if (v > best_val || (v == best_val && idx < best_idx)) {
            best_val = v;
            best_idx = idx;
        }
    }

    best_vals[tid] = best_val;
    best_idxs[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const float other_val = best_vals[tid + stride];
            const uint other_idx = best_idxs[tid + stride];
            const float cur_val = best_vals[tid];
            const uint cur_idx = best_idxs[tid];
            if (other_val > cur_val || (other_val == cur_val && other_idx < cur_idx)) {
                best_vals[tid] = other_val;
                best_idxs[tid] = other_idx;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        out[0] = best_idxs[0];
        out[1] = as_type<uint>(best_vals[0]);
    }
}
