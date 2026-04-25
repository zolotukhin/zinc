# Effort 10 — Qwen 3.6 35B-A3B decode + prefill speedups on RDNA4

## Current standing  (cycle 20, run 1)

```
baseline (cycle 0):     24.01 tok/s decode  (BW util 60.3%)
best   (cycle 13):      25.36 tok/s decode  (BW util 63.7%)   +5.6%
post-cleanup (c20):     25.40 tok/s decode  (BW util 63.7%)
```

## ⚠ Baseline correction  (post-reboot, 2026-04-25)

The first 20 cycles were measured against an **R9700 stuck in low-DPM
state** after 22 days uptime + 8 days of continuous llama-server
holding GPU resources. The reboot restored the GPU to peak clocks
(~108 tok/s on llama-bench tg128 for Qwen 3.6 35B-A3B, matching the
historical 107 tok/s baseline from blog posts). Implications:

- ZINC's relative position vs llama.cpp is **much worse** than the
  pre-reboot data suggested. On Qwen 3.6 35B-A3B:
  - ZINC decode 29 tok/s vs **llama-bench tg128 = 108.7** (= **27% of
    peak**, not the 90% the previous run inferred).
  - ZINC prefill 26 tok/s vs **llama-server core = 54** (= **48%**) and
    vs llama-bench pp128 = 1159 burst (= **2%**, but burst pp is not the
    realistic comparison).
- The +5.6% the loop banked over 20 cycles is real, but happened
  against a degraded baseline. Reruns of the agent's wins (cycles 4, 8,
  13) on the post-reboot baseline should reproduce the same shaders'
  benefit; the loop should NOT discard those commits.
- Baseline-mode comparison (server config: `--parallel 4 -ctk q8_0
  -ctv q8_0 -b 4096 -ub 1024 --flash-attn on`) reports **30.4 tok/s
  decode** on the post-reboot llama-server. ZINC at 29 vs 30 is 95% of
  *that* baseline — the published comparison still looks favorable in
  the perf JSON, but it's measuring a llama config that itself runs
  ~28% of llama-bench's bare-metal peak. Don't take the 95% as a sign
  the work is done.

3 wins kept across 20 cycles. The pattern that delivers: **fuse RMS norm
into the adjacent DMMV that consumes it.** Two such shaders shipped.
Five other patterns measured-flat or measured-negative (catalogued below
under known flat).

## Wins that landed

| cycle | tok/s | Δ | what |
|---|---:|---:|---|
| 4  | 24.19 | +0.19 | Register-cache `delta_net` state across decay/dot/update/readout in `ssm_delta_net.comp`. Eliminates redundant state-buffer touches. |
| 8  | 24.79 | +0.61 | New `rms_norm_dmmv_f32.comp` shader — fuses per-MoE-layer `rms_norm_mul → router DMMV` into one dispatch on architectures with f32 router weights (Qwen 3.5/3.6). |
| 13 | 25.36 | +0.57 | New `rms_norm_dmmv_q4k_alpha_beta.comp` shader — fuses (RMS norm → alpha DMMV → beta DMMV) trio in the SSM proj path into one dispatch. Same pattern as cycle 8 but Q4_K weights and two outputs sharing one super-block decode. |

Plus 2 foundation-keeps: cycle 16 narrowed cycle 13's post-fused-RMS
barrier to buffer-scoped, cycle 20 cleaned up dead `rms_norm_dmmv_q8_0_kv`
infra leftover from cycle 14's revert.

## Where the time goes — post-cycle-20

```
GPU decode token = 37.18 ms  (= 26.9 tok/s, profile-mode)
  attention =  4.65 ms  (12%)   over 10 attention layers
  ssm       = 14.85 ms  (40%)   over 30 SSM layers
    proj    =  7.92 ms  (now: cycle-13 fused alpha+beta + 2 unfused DMMVs)
    conv    =  0.42 ms
    delta   =  3.40 ms  (cycle-4 register cache helped)
    gnorm   =  0.34 ms
    out     =  3.10 ms  (untouched)
  moe       = 12.40 ms  (33%)
    router  =  1.40 ms  (cycle 8 fused: was 2.20, now combined with FFN-norm)
    topk    =  0.57 ms
    gate_up =  4.66 ms  (kpar already enabled — multiple wide attempts flat)
    swiglu  =  0.31 ms
    down    =  4.67 ms  (kpar already enabled)
    acc     =  0.52 ms
  tail      =  0.96 ms  (LM head wide-NUM_ROWS=8 from earlier session)
  embed     =  0.01 ms
```

Active params per token ≈ 3 GB. Bandwidth ceiling on R9700: ~5.2 ms.
Still at ~7× the ceiling, so dispatch / sync overhead remains the
dominant cost.

## Big levers (current)

### A. More fused-RMS+DMMV shaders  — proven pattern, candidates below

Cycles 8 and 13 each shipped ~+0.6 tok/s by fusing one `rms_norm_mul`
with the immediately-consuming DMMV. The hot path still has several
RMS-norm → DMMV pairs that haven't been fused:

- `attn_norm → wqkv` (the SSM proj's biggest DMMV, M=conv_channels≈6144).
  Cycle 10 attempted this and reverted — flag-on gain fell short of
  checkpoint due to redundant per-WG RMS work. The reason it failed: the
  shader recomputes RMS reduction for every output row. Fix: precompute
  RMS once per block via shared memory, then reuse across all NUM_ROWS
  rows. Re-attempt with that structure.
- `attn_norm → ssm_z` (the d_inner DMMV, M≈4096). Same pattern; same
  fix.
- `attn_norm → attention Q proj` (M=2048, K=2048). Different from KV
  fusion (cycle 14 attempted; reverted) — Q is single-output and the
  caller can use the existing `rms_norm_dmmv_f32` shape.
- `ffn_norm → MoE router` is what cycle 8 already shipped. Don't redo.

### B. Cross-token batched MoE FFN  — infrastructure shipped, not wired

Shader `dmmv_q4k_moe_batched.comp` and `recordMoeBatchedDispatch` helper
landed in commit `c36bd23`. Pipeline loads at startup. **No call site
uses it.** This is the prefill lever — won't help the current decode
benchmark, but the infra is ready.

To wire in: phase 1.1 in the original plan still applies — build N-token
routing buffer, allocate `[N × n_experts_used × inter]` output scratch,
dispatch new shader for gate / up / down, scatter-accumulate back into
hidden, relax `canUseBatchedPrefillRdna` for qwen35moe with per-layer-
type detection.

### C. Eliminate dispatches entirely  — best decode lever now

Once a sync-bound model is at 7× the BW ceiling, the only thing that
moves the needle is reducing the **number** of dispatches. Two specific
candidates the loop hasn't tried:

- Fused **`ssm_out` + FFN-RMS-norm** shader. The SSM tail does
  `ssm_out DMMV → residual add → ffn_norm RMS norm`; this is the same
  shape as `rms_norm_add` (already exists for Gemma post_ffw_norm, see
  commit `a5f1fdc`) but with a Q4_K DMMV in front. Eliminates 1 dispatch
  + 1 barrier per SSM layer × 30 = 30 dispatches saved.
- **KV-cache-write fused into K-projection** for attention layers.
  Saves 10 dispatches + 10 barriers per token. Distinct from cycle 14's
  failed fused-RMS+K+V — that one tried to fuse three things; this one
  fuses the K projection's output directly into the cache write at end
  of the kernel (skipping the intermediate k_buf round-trip).

## Known flat — captured by the loop, do not re-attempt

- **Q4_K × Q8_1 mmq for SSM proj GEMV** (commit 27f0c76 + 3fef46e). No
  speedup on Qwen 3.6 (SSM phase 15.94 ms either way). Bandwidth-bound
  on weight side; mmq cuts activation+compute, neither is the
  bottleneck. Re-attempt only in a GEMM context.
- **Alpha+beta SSM proj fusion via `dmmv_q4k_fused_gate_up`** (3fef46e).
  The four SSM proj DMMVs already overlap on RDNA4 (no inter-DMMV
  barriers). Different from cycle 13's fused-RMS+alpha+beta which DID
  win because it added the RMS norm into the same dispatch.
- **Dense fused gate+up** (`dmmv_q4k_fused_gate_up.comp`, 339c886).
  Regresses Gemma 4 31B by +11% from doubled per-WG register pressure
  on wide inter_dim=25600.
- **NUM_ROWS=4 medium variant of dmmv_q4k for SSM out** (M=2048).
  Already saturating occupancy with NUM_ROWS=2 → 1024 WGs.
- **NUM_ROWS=4 wide kpar variant of dmmv_q4k_moe_kpar for MoE
  gate/up** (cycle 12). -0.05 tok/s; gate_up phase +0.10 ms vs baseline.
- **Q8_0 wide NUM_ROWS=4 + register-tiled activation reads for SSM proj
  wqkv** (cycle 11). Both NUM_ROWS=4 (1 wave) and NUM_ROWS=2 with hoisted
  xv variants neutral.
- **Attn-RMS + Q8_0/Q4_K SSM proj wqkv DMMV fusion** (cycle 10). Small
  flag-on gain that fell short of checkpoint due to redundant per-WG
  RMS work. Re-attempt only with shared-memory RMS reduction (see
  lever A).
- **Fused RMS + Q8_0 attention K+V DMMVs** (cycle 14). Reverted — three-
  way fusion has too much register pressure. Two-way KV+RMS might still
  work; cycle didn't try it.
- **f16-quantized MoE router weight** (cycle 17). Per-layer device-local
  f16 buffers built from f32 ffn_gate_inp at engine init. Negative.
- **Barrier scoping** (cycles 1, 16, 18). Narrowing
  `computeBarrier()` → `computeBufferBarrier()` does not move the metric
  on RADV; the runtime apparently doesn't differentiate the access masks
  the way the test hypothesis assumed.
- **MoE down + SwiGLU fusion** (cycle 9). Both kpar+swiglu and triple-
  fused down+swiglu+acc variants regressed.
- **Q4_K MoE forward gate+up+SwiGLU fusion** (cycle 19). +0.12% — below
  noise floor. Reverted.
- **Register-cache `delta_net_output` across the two passes in
  `ssm_gated_norm.comp`** (cycle 5). The pass-2 global re-read isn't
  the bottleneck.
- **Cycle 6/7's fused MoE down+weighted_acc shader** (Q4_K + Q5_K).
  +0.16 vs checkpoint, below override threshold.

## Gaps the loop hasn't entered

1. **Prefill mode is not on the loop's metric.** Effort 10 spec is
   `metricMode: "decode"`. The 24.5 → 54.5 tok/s prefill gap to
   llama.cpp is 2.2× larger than the decode gap and hasn't been chased.
   The cross-token batched MoE shader (lever B above) only pays off
   under prefill mode. Either flip the spec to `prefill` for a parallel
   run or land a separate `--effort 11` with the prefill spec.
2. **Attention path is essentially untouched.** All 20 cycles focused on
   SSM and MoE. Attention is 4.65 ms (12%); not the biggest chunk, but
   no fusion candidates from the attention RMS norm have been tried.
3. **The dense `rms_norm_add` shader from earlier session** (Gemma's
   `post_ffw_norm` fold-in) hasn't been considered for SSM out + FFN-norm
   fusion on Qwen 3.6, even though it's the same shape (RMS norm of
   one buffer accumulated into another).
4. **VkCmdPipelineBarrier2 with explicit accessMask precision** — listed
   in the loop's nextIdeas at cycle 1, never actually attempted. Could
   move metric on RADV if cycle 1's hypothesis ("RADV doesn't
   differentiate") was wrong about the mechanism (vs the implementation).
5. **NUM_COLS-specialized dense matmul for prefill** (the llama.cpp
   pattern from Effort 6's structuralSwingIdeas). Generally a prefill
   lever, not decode.

## Recommended next moves for the loop

1. (Highest ROI on decode) Lever A re-attempt with shared-memory RMS
   reduction: fuse `attn_norm → wqkv` in the SSM proj. Apply the same
   precompute-once-then-reuse pattern across NUM_ROWS rows that makes
   the cycle 13 shader work.
2. (Equal ROI) Lever C #1: fused `ssm_out` + FFN-RMS-norm shader.
   Mirrors `rms_norm_add` shape exactly but with Q4_K DMMV preceding.
3. (Equal ROI) Lever C #2: KV-cache-write fused into K-projection on
   attention layers.
4. After 1-3 land or all measure flat: pivot to lever B (cross-token
   batched MoE FFN). Wire the existing `dmmv_q4k_moe_batched.comp` into
   `prefillBatched`. Requires switching the loop's metric mode to
   prefill OR running a parallel effort.

## Constraints

- Coherence on all 6 RDNA catalog models for every accepted change.
  Validate via `ZINC_BATCHED_PREFILL=validate` for prefill changes and
  golden output text for decode changes.
- Don't regress Gemma 4 31B prefill (40.5 tok/s) / decode (4.6 tok/s)
  baselines or Qwen3-8B prefill (199 tok/s) / decode (66 tok/s) baselines
  established earlier this session.
- Loop's noise threshold is +0.25 tok/s. Cycles below that get rejected
  even if a clean structural improvement.
