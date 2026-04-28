#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint k;
    uint gate_offset;
    uint gate_type; // 0=f32, 1=f16, 2=q8_0
};

static inline float load_gate_weight(device const uchar* W, constant Params& p, uint idx) {
    if (p.gate_type == 0u) {
        device const float* wf = (device const float*)(W + p.gate_offset);
        return wf[idx];
    }
    if (p.gate_type == 1u) {
        device const half* wh = (device const half*)(W + p.gate_offset);
        return float(wh[idx]);
    }

    const uint block = idx >> 5;
    const uint lane = idx & 31u;
    device const uchar* b = W + p.gate_offset + block * 34u;
    const float d = float(as_type<half>(*(device const ushort*)(b)));
    const char q = as_type<char>(b[2u + lane]);
    return d * float(q);
}

kernel void main0(
    device float* accum [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device const uchar* gate_weight [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partials[256];

    float dot = 0.0f;
    for (uint i = tid; i < p.k; i += 256u) {
        dot = fma(load_gate_weight(gate_weight, p, i), input[i], dot);
    }
    partials[tid] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float gate = 1.0f / (1.0f + exp(-partials[0]));
    for (uint i = tid; i < p.n; i += 256u) {
        accum[i] += gate * src[i];
    }
}
