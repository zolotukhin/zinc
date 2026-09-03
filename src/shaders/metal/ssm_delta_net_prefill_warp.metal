// ssm_delta_net_prefill_warp.metal — warp-level delta-net scan (ZERO barriers in hot loop)
//
// Port of the CUDA ssm_delta_net_warp approach to Metal:
//   Each SIMDGROUP (32 lanes) owns one ROW of the state matrix.
//   Each lane holds 4 state elements (cols_per_lane = head_v_dim / simd_width = 128/32 = 4).
//   ALL reductions use simd_sum() — NO threadgroup_barrier per token iteration.
//
// FIXED (2026-06-30): swap row/col ownership — warp owns ROW (not column)
// to compute S@k (row-wise dot product), matching the block shader.
//
// Compared to ssm_delta_net_prefill.metal: eliminates 4 barriers/iteration → 0.
// The only barrier is ONE before the scan loop (for gate/beta precompute).
//
// Grid: (dt_rank, head_v_dim / n_simdgroups) threadgroups
// Threadgroup: 32 * n_simdgroups = 128 threads
//
// Shape-generic in dt_rank / n_group / d_inner via the push constants (the
// Qwen3.6 35B-A3B dt_rank=32 and the Qwen3.6/3.8 27B dt_rank=48 both route
// here); requires head_v_dim == d_state == 128 (state rows are 4 columns per
// lane) and n_tokens <= 512 (gate/beta threadgroup cache). The dispatcher
// (`canUseSsmDeltaNetPrefillWarp`) enforces both.

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint d_inner;
    uint dt_rank;
    uint head_v_dim;
    uint d_state;
    uint n_group;
    uint has_dt_bias;
    uint has_ssm_a;
    uint n_tokens;
    uint alpha_stride;
    uint beta_stride;
    uint conv_stride;
    uint output_stride;
    uint alpha_offset;
    uint beta_offset;
    uint conv_offset;
    uint output_offset;
};

// Register-resident delta-net scan; head_v_dim == d_state == 128 (see header).
kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* conv_out [[buffer(1)]],
    device const float* alpha [[buffer(2)]],
    device const float* dt_bias [[buffer(3)]],
    device const float* ssm_a [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    device float* state [[buffer(6)]],
    device float* output [[buffer(7)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint simd_width [[thread_execution_width]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    // MSL requires all position attributes to share a dimensionality, so the
    // thread index arrives as uint3 alongside the 2-D threadgroup position.
    const uint tid = tid3.x;
    const uint head = tg_pos.x;
    const uint simd_lane = tid % simd_width;
    const uint simd_idx = tid / simd_width;
    const uint row = tg_pos.y * simdgroups_per_tg + simd_idx;

    if (head >= p.dt_rank || row >= p.head_v_dim || p.head_v_dim != 128u || p.d_state != 128u || p.n_tokens > 512u) {
        return;
    }

    constexpr uint hv = 128u;
    constexpr uint cols_per_lane = 4u;  // hv / 32 = 4

    const uint qk_dim = p.d_state * p.n_group;
    const uint k_len = min(p.d_state, p.head_v_dim);
    const uint k_hi = (p.n_group == p.dt_rank) ? head : (head % p.n_group);
    const uint head_state_base = head * hv * hv;

    // Load initial state: each lane owns 4 elements at cols simd_lane, simd_lane+32, simd_lane+64, simd_lane+96
    float s_shard[cols_per_lane];
    for (uint r = 0u; r < cols_per_lane; ++r) {
        const uint col = r * simd_width + simd_lane;
        s_shard[r] = state[head_state_base + row * hv + col];
    }

    // Precompute per-head constants
    const float dt_bias_val = (p.has_dt_bias != 0u) ? dt_bias[head] : 0.0f;
    const float ssm_a_val = (p.has_ssm_a != 0u) ? ssm_a[head] : 0.0f;
    const float inv_sqrt_d_state = 1.0f / sqrt(float(p.d_state));

    // Precompute gate[t] and beta[t] for ALL tokens — parallel, no state dependency.
    // Stored in threadgroup memory (2 * n_tokens floats).
    threadgroup float sh_gate[512];  // max n_tokens = 512
    threadgroup float sh_beta[512];
    const uint tg_threads = simd_width * simdgroups_per_tg;
    for (uint t = tid; t < p.n_tokens; t += tg_threads) {
        const float a = alpha[p.alpha_offset + t * p.alpha_stride + head] + dt_bias_val;
        const float sp = log(1.0f + fast::exp(a));
        sh_gate[t] = (p.has_ssm_a != 0u) ? fast::exp(sp * ssm_a_val) : fast::exp(-sp);
        sh_beta[t] = fast::divide(1.0f, 1.0f + fast::exp(-beta[p.beta_offset + t * p.beta_stride + head]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // ONE barrier before the loop

    // Scan loop: ZERO threadgroup_barrier calls — all reductions via simd_sum
    for (uint t = 0u; t < p.n_tokens; ++t) {
        const uint conv_token_base = p.conv_offset + t * p.conv_stride;
        const uint q_base = conv_token_base + k_hi * p.d_state;
        const uint k_base = conv_token_base + qk_dim + k_hi * p.d_state;
        const uint v_base = conv_token_base + 2u * qk_dim + head * hv;

        // Load Q, K into registers (4 per lane, indexed by COLUMN)
        float q_reg[cols_per_lane], k_reg[cols_per_lane];
        float sumq = 0.0f, sumk = 0.0f;
        for (uint r = 0u; r < cols_per_lane; ++r) {
            const uint col_idx = r * simd_width + simd_lane;
            if (col_idx < k_len) {
                q_reg[r] = conv_out[q_base + col_idx];
                k_reg[r] = conv_out[k_base + col_idx];
                sumq = fma(q_reg[r], q_reg[r], sumq);
                sumk = fma(k_reg[r], k_reg[r], sumk);
            } else {
                q_reg[r] = 0.0f;
                k_reg[r] = 0.0f;
            }
        }

        // Q/K norms: simd_sum (no barrier — each simdgroup sums all 128 values internally)
        const float q_norm_sq = simd_sum(sumq);
        const float k_norm_sq = simd_sum(sumk);
        const float q_rinv = fast::rsqrt(fast::max(q_norm_sq, 1.0e-13f)) * inv_sqrt_d_state;
        const float k_rinv = fast::rsqrt(fast::max(k_norm_sq, 1.0e-13f));

        // Gate and beta: read from precomputed threadgroup memory
        const float gate = sh_gate[t];
        const float b_val = sh_beta[t];

        // Load V for this row (indexed by ROW, not column)
        const float v_val = conv_out[v_base + row];

        // State update: sk = (S @ k_norm)[row] — simd_sum across COLUMNS
        float sk_partial = 0.0f;
        for (uint r = 0u; r < cols_per_lane; ++r) {
            s_shard[r] *= gate;
            sk_partial = fma(s_shard[r], k_reg[r] * k_rinv, sk_partial);
        }
        const float sk = simd_sum(sk_partial);
        const float d = b_val * (v_val - sk);

        // Readout: o = (S @ q_norm)[row] — simd_sum across COLUMNS
        float o_partial = 0.0f;
        for (uint r = 0u; r < cols_per_lane; ++r) {
            s_shard[r] = fma(k_reg[r] * k_rinv, d, s_shard[r]);  // rank-1 update
            o_partial = fma(s_shard[r], q_reg[r] * q_rinv, o_partial);
        }
        const float o = simd_sum(o_partial);

        // Write output (lane 0 of each simdgroup writes)
        if (simd_lane == 0u) {
            output[p.output_offset + t * p.output_stride + head * hv + row] = o;
        }
    }

    // Write final state back to global memory
    for (uint r = 0u; r < cols_per_lane; ++r) {
        const uint col = r * simd_width + simd_lane;
        state[head_state_base + row * hv + col] = s_shard[r];
    }
}
