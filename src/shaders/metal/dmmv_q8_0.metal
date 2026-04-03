#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Native Metal Q8_0 DMMV.
//
// Q8_0 is the dominant path for router, shared-expert, and SSM projections on
// the target Qwen3.5-35B-A3B model. The old SPIRV-Cross shader only processed
// two rows per workgroup and re-read the same X vector from device memory for
// each row. On Apple Silicon the shared/unified memory path benefits from
// staging X once per workgroup and letting multiple simdgroups reuse it.
//
// One simdgroup (32 lanes) handles one output row. Each lane owns a subset of
// the 32-element Q8_0 blocks for that row and accumulates 8 packed int8 dot
// products per block. The workgroup can therefore process 2 rows with a 64-wide
// launch or 8 rows with the 256-wide launch used on the hot decode path.
//
// Q8_0 block layout (34 bytes, 32 elements):
//   [0..1]   d  (float16) scale
//   [2..33]  qs (32 x int8)

kernel void main0(
    constant DmmvPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    // Staging activations as half4 cuts threadgroup memory in half versus
    // float4 while keeping the final accumulation in float. On Apple GPUs the
    // q8 decode path is occupancy-sensitive; reducing the staged footprint lets
    // the wider launch shapes stay resident more consistently.
    threadgroup half4 x_cache4[1024];

    const uint tg_size = simdgroups_per_tg * 32u;
    device const float* input = X + (p.x_offset >> 2);
    device float* output = Y + (p.y_offset >> 2);

    const uint k_vec4 = p.K >> 2;
    for (uint i = local_id; i < k_vec4; i += tg_size) {
        x_cache4[i] = half4(*(device const float4*)(input + (i << 2)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tg_id * simdgroups_per_tg + sg_idx;
    if (row >= p.M) return;

    const uint blocks_per_row = p.K >> 5;
    device const uchar* row_ptr = W + p.a_offset + ulong(row) * ulong(blocks_per_row) * 34ull;

    float acc = 0.0f;
    for (uint bi = lane; bi < blocks_per_row; bi += 32u) {
        device const uchar* block = row_ptr + bi * 34u;
        const float scale = float(as_type<half>(*(device const ushort*)(block)));
        device const packed_char4* quants = (device const packed_char4*)(block + 2u);
        const uint x_base = bi * 8u;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const char4 q = char4(quants[vi]);
            const half4 x = x_cache4[x_base + vi];
            const half4 q_half = half4(q);
            acc = fma(scale, float(dot(q_half, x)), acc);
        }
    }

    const float sum = simd_sum(acc);
    if (lane == 0u) {
        output[row] = sum;
    }
}
