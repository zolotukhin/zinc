#include <metal_stdlib>
using namespace metal;

// Fused Gemma MoE gate/up Q4_K DMMV + GeGLU for single-token decode.
//
// This is the same routed small-batch shape as dmmv_q4k_moe_gate_up, but it
// applies Gemma's GeGLU immediately and writes the activated expert output.

struct MoeGateUpDmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint expert_stride;
    uint gate_base_offset;
    uint up_base_offset;
    uint x_expert_stride;
    uint x_offset;
    uint gate_y_offset;
    uint up_y_offset;
};

inline float2 get_scale_min_k4(uint j, device const uchar* sc) {
    if (j < 4) {
        return float2(float(sc[j] & 63), float(sc[j + 4] & 63));
    }
    return float2(
        float((sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4)),
        float(((sc[j + 4] >> 4) & 0x0F) | ((sc[j] >> 6) << 4))
    );
}

inline void accumulate_q4k(
    device const uchar* block,
    threadgroup float4* x_cache4,
    uint bi,
    uint lane,
    thread float& acc
) {
    const float d = float(as_type<half>(*(device const ushort*)(block)));
    const float dmin = float(as_type<half>(*(device const ushort*)(block + 2)));
    device const uchar* scales = block + 4;
    device const uchar* quants = block + 16;

    const uint byte_off = lane * 4u;
    const uint j = byte_off / 32u;
    const uint local_off = byte_off % 32u;

    const uchar4 qbytes = *(device const uchar4*)(quants + byte_off);
    const float2 sm_lo = get_scale_min_k4(j * 2u, scales);
    const float2 sm_hi = get_scale_min_k4(j * 2u + 1u, scales);

    const float d_sc_lo = d * sm_lo.x;
    const float d_m_lo = dmin * sm_lo.y;
    const float d_sc_hi = d * sm_hi.x;
    const float d_m_hi = dmin * sm_hi.y;

    const uint col_lo = bi * 256u + j * 64u + local_off;
    const uint col_hi = col_lo + 32u;

    const float4 x_lo = x_cache4[col_lo >> 2];
    const float4 x_hi = x_cache4[col_hi >> 2];

    const uchar4 q_lo = uchar4(
        qbytes.x & 0x0F,
        qbytes.y & 0x0F,
        qbytes.z & 0x0F,
        qbytes.w & 0x0F
    );
    const uchar4 q_hi = uchar4(
        qbytes.x >> 4,
        qbytes.y >> 4,
        qbytes.z >> 4,
        qbytes.w >> 4
    );

    const float4 lo_vals = fma(float4(q_lo), float4(d_sc_lo), float4(-d_m_lo));
    const float4 hi_vals = fma(float4(q_hi), float4(d_sc_hi), float4(-d_m_hi));

    acc += dot(lo_vals, x_lo) + dot(hi_vals, x_hi);
}

inline float geglu(float gate, float up) {
    const float g3 = gate * gate * gate;
    float inner = 0.7978845608f * (gate + 0.044715f * g3);
    inner = clamp(inner, -15.0f, 15.0f);
    const float gelu_gate = 0.5f * gate * (1.0f + precise::tanh(inner));
    return gelu_gate * up;
}

#define TG_SIZE 256
#define ROWS_PER_SIMDGROUP 2
#define ROWS_PER_TG ((TG_SIZE / 32) * ROWS_PER_SIMDGROUP)

kernel void main0(
    device const uchar* W                         [[buffer(0)]],
    constant MoeGateUpDmmvPush& p                 [[buffer(1)]],
    device const float* X                         [[buffer(2)]],
    device float* activatedY                      [[buffer(3)]],
    device const uint* expert_ids                 [[buffer(4)]],
    uint3 tg_pos                                  [[threadgroup_position_in_grid]],
    uint3 local_pos                               [[thread_position_in_threadgroup]]
) {
    const uint expert_slot = tg_pos.y;
    const uint expert_id = expert_ids[expert_slot];
    device const float* input = X + (p.x_offset / 4u) + expert_slot * p.x_expert_stride;
    threadgroup float4 x_cache4[1024];

    const uint local_id = local_pos.x;
    const uint sg_idx = local_id / 32u;
    const uint lane = local_id % 32u;

    const uint k_vec4 = p.K >> 2;
    for (uint i = local_id; i < k_vec4; i += TG_SIZE) {
        x_cache4[i] = *(device const float4*)(input + (i << 2));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row0 = tg_pos.x * ROWS_PER_TG + sg_idx * ROWS_PER_SIMDGROUP;
    const uint row1 = row0 + 1u;
    const bool valid0 = row0 < p.M;
    const bool valid1 = row1 < p.M;
    if (!valid0 && !valid1) {
        return;
    }

    const uint bpr = p.K / 256u;
    const ulong row_bytes = ulong(bpr) * 144ul;
    const ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);

    device const uchar* gate_row0 = W + expert_base + ulong(p.gate_base_offset) + ulong(row0) * row_bytes;
    device const uchar* gate_row1 = W + expert_base + ulong(p.gate_base_offset) + ulong(row1) * row_bytes;
    device const uchar* up_row0 = W + expert_base + ulong(p.up_base_offset) + ulong(row0) * row_bytes;
    device const uchar* up_row1 = W + expert_base + ulong(p.up_base_offset) + ulong(row1) * row_bytes;

    float gate_acc0 = 0.0f;
    float gate_acc1 = 0.0f;
    float up_acc0 = 0.0f;
    float up_acc1 = 0.0f;

    for (uint bi = 0u; bi < bpr; bi++) {
        if (valid0) {
            accumulate_q4k(gate_row0 + ulong(bi) * 144ul, x_cache4, bi, lane, gate_acc0);
            accumulate_q4k(up_row0 + ulong(bi) * 144ul, x_cache4, bi, lane, up_acc0);
        }
        if (valid1) {
            accumulate_q4k(gate_row1 + ulong(bi) * 144ul, x_cache4, bi, lane, gate_acc1);
            accumulate_q4k(up_row1 + ulong(bi) * 144ul, x_cache4, bi, lane, up_acc1);
        }
    }

    const float gate_sum0 = simd_sum(gate_acc0);
    const float gate_sum1 = simd_sum(gate_acc1);
    const float up_sum0 = simd_sum(up_acc0);
    const float up_sum1 = simd_sum(up_acc1);

    if (lane == 0u) {
        device float* out = activatedY + (p.gate_y_offset / 4u) + expert_slot * p.M;
        if (valid0) {
            out[row0] = geglu(gate_sum0, up_sum0);
        }
        if (valid1) {
            out[row1] = geglu(gate_sum1, up_sum1);
        }
    }
}
