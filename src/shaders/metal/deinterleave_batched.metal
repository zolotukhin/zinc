#include <metal_stdlib>
using namespace metal;

struct Push {
    uint head_dim;
    uint n_heads;
    uint n_tokens;
};

kernel void main0(
    constant Push & push [[buffer(0)]],
    device float * q_out [[buffer(1)]],
    device const float * q_gate [[buffer(2)]],
    device float * gate_out [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint per_token = push.head_dim * push.n_heads;
    const uint total = per_token * push.n_tokens;
    if (gid >= total) return;

    const uint token = gid / per_token;
    const uint local = gid - token * per_token;
    const uint head = local / push.head_dim;
    const uint dim = local - head * push.head_dim;
    const uint in_base = token * per_token * 2 + head * push.head_dim * 2 + dim;

    q_out[gid] = q_gate[in_base];
    gate_out[gid] = q_gate[in_base + push.head_dim];
}
