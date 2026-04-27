#include <metal_stdlib>
using namespace metal;

struct DualQ8DmmvPush {
    uint M0;
    uint M1;
    uint K;
    uint a0_offset;
    uint a1_offset;
    uint x_offset;
    uint y0_offset;
    uint y1_offset;
};

static inline float gelu_tanh(float g) {
    const float g3 = g * g * g;
    float inner = 0.7978845608f * (g + 0.044715f * g3);
    inner = clamp(inner, -15.0f, 15.0f);
    return 0.5f * g * (1.0f + precise::tanh(inner));
}

// Paired equal-shape Q8_0 DMMV for Gemma shared expert gate/up.
// Each simdgroup computes two rows from the gate and up matrices, then writes
// GeGLU(gate) * up directly to the shared activation buffer.
kernel void main0(
    constant DualQ8DmmvPush& p [[buffer(0)]],
    device const uchar* WGate [[buffer(1)]],
    device const uchar* WUp [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* Y [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    const uint base_row = (tg_id * simdgroups_per_tg + sg_idx) * 2u;
    if (base_row >= p.M0 || base_row >= p.M1) return;

    device const float* input = X + (p.x_offset >> 2);
    device float* output = Y + (p.y0_offset >> 2);

    const uint blocks_per_row = p.K >> 5;
    const ulong row_bytes = ulong(blocks_per_row) * 34ull;
    const bool has_next = base_row + 1u < p.M0 && base_row + 1u < p.M1;
    device const uchar* gate0 = WGate + p.a0_offset + ulong(base_row) * row_bytes;
    device const uchar* gate1 = has_next ? (gate0 + row_bytes) : gate0;
    device const uchar* up0 = WUp + p.a1_offset + ulong(base_row) * row_bytes;
    device const uchar* up1 = has_next ? (up0 + row_bytes) : up0;

    float gate_acc0 = 0.0f;
    float gate_acc1 = 0.0f;
    float up_acc0 = 0.0f;
    float up_acc1 = 0.0f;

    for (uint bi = lane; bi < blocks_per_row; bi += 32u) {
        device const uchar* gate_blk0 = gate0 + bi * 34u;
        device const uchar* gate_blk1 = gate1 + bi * 34u;
        device const uchar* up_blk0 = up0 + bi * 34u;
        device const uchar* up_blk1 = up1 + bi * 34u;
        const float gate_s0 = float(as_type<half>(*(device const ushort*)(gate_blk0)));
        const float gate_s1 = float(as_type<half>(*(device const ushort*)(gate_blk1)));
        const float up_s0 = float(as_type<half>(*(device const ushort*)(up_blk0)));
        const float up_s1 = float(as_type<half>(*(device const ushort*)(up_blk1)));
        device const packed_char4* gate_q0 = (device const packed_char4*)(gate_blk0 + 2u);
        device const packed_char4* gate_q1 = (device const packed_char4*)(gate_blk1 + 2u);
        device const packed_char4* up_q0 = (device const packed_char4*)(up_blk0 + 2u);
        device const packed_char4* up_q1 = (device const packed_char4*)(up_blk1 + 2u);
        const uint x_base = bi << 5;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const float4 x = *(device const float4*)(input + x_base + (vi << 2));
            gate_acc0 = fma(gate_s0, dot(float4(char4(gate_q0[vi])), x), gate_acc0);
            gate_acc1 = fma(gate_s1, dot(float4(char4(gate_q1[vi])), x), gate_acc1);
            up_acc0 = fma(up_s0, dot(float4(char4(up_q0[vi])), x), up_acc0);
            up_acc1 = fma(up_s1, dot(float4(char4(up_q1[vi])), x), up_acc1);
        }
    }

    const float gate_sum0 = simd_sum(gate_acc0);
    const float up_sum0 = simd_sum(up_acc0);
    if (lane == 0u) {
        output[base_row] = gelu_tanh(gate_sum0) * up_sum0;
    }

    if (has_next) {
        const float gate_sum1 = simd_sum(gate_acc1);
        const float up_sum1 = simd_sum(up_acc1);
        if (lane == 0u) {
            output[base_row + 1u] = gelu_tanh(gate_sum1) * up_sum1;
        }
    }
}
