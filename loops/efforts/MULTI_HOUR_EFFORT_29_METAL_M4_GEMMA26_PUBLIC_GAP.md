# Effort 29 — Gemma4 26B-A4B public-suite gap (Metal, M4 Max) — STATUS: OPEN

Close the remaining public Metal-suite gap vs llama.cpp on
`gemma4-26b-a4b-q4k-m` (chat mode), decode-primary with prefill guarded.
Public site rows (Metal target, generated 2026-06-13, still current):
prefill 334.8 vs 411.78 (81%), decode 69.43 vs 83.49 (83%), end-to-end
19.36 vs 116.14 (16.7%, 5.9x total-latency gap — likely cold
pipeline/load/setup; needs measurement coverage since the harness can only
promote prefill/decode). One fresh run (2026-06-15,
`.metal_optimize/2026-06-15T04-45-33`) was stopped manually at cycle 12 amid
Q8 launch-shape churn and a persistent prefill-guard regression.
**Nothing from that run was merged** — its cycle commits are dangling, off
every branch; do not assume any of its decode changes exist at HEAD, and do
not resume that run state or `--resume` Effort 11.

## Landed

Nothing from this effort. Live foundations from Effort 11 (verified at HEAD,
`src/compute/forward_metal.zig`): structural batched-route-pack Gemma MoE
prefill (`prefill actual path: batched-route-pack ... structural_batched=yes`,
route-pack occupancy profiling) and
`gemma_moe_weighted_post_norm_residual{,_batched}` pipelines (prefill
finalizer). Note: `dmmv_q8_0_pair_k2816.metal` and `dmmv_q8_0_argmax.metal`
exist as shader files on main but are currently UNWIRED (no pipeline loads
them); the stopped run's decode wiring for them was never merged.

## Dead ends (do not retry)

- Decode Q8 threadgroup/launch-shape retunes (shared gate/up K=2816 128-thread, shared-down 128-thread selector, LM-head argmax 1024-thread gate) → no stable decode lift over 11 cycles (67.2 → 68.5 best, +2%).
- Q8_0 LM-head partial argmax for greedy decode → only material signal and still just ~+0.4 tok/s.
- Packed `dmmv_q8_0_pair_k2816` reduction/writeback tail → rejected in-run.
- Weighted post-norm finalizer gating toggles (decode-on, prefill opt-in/opt-out/exact-shape restore) → neither direction recovered the prefill guard, which oscillated 320–334 vs the 355.6 warm cycle-1 baseline (−6.4%, violating the −3% guard).
- From Effort 11, still binding: Q5_1 exact-6 tails, Q4_K exact-4 tails, broad route-pack alt block widths, routine guard audits after `structural_batched=yes`, Gemma GPU LM-head attempts (regressed total time), broad Q8 threadgroup/paired variants.

## Still open

1. Harness policy first: make a cross-effort guard regression beyond −3%
   revert the offending candidate or stop the run; the stopped run kept
   neutral changes while the guard was red.
2. Re-measure a clean no-agent baseline after cool-down. If prefill is
   ~333–335, treat cycle 1's 355.6 guard as a warm outlier and reset the guard
   baseline; if it returns to 355+, start from a clean tree.
3. Decode (primary, target 84): not another Q8 launch retune. Use profile
   evidence to remove a named barrier/dispatch family in attention or Gemma
   MoE (last profile: 537 dispatches / 361 barriers per step; hot Q8 shapes
   lm-head M=262144 K=2816, shared gate/up M=2112 K=2816 x1920 calls), or if
   routed MoE dominates, evaluate grouped expert dispatch for one-token decode.
4. End-to-end latency: add measurement for pipeline compilation, model load,
   and first-request setup — the 5.9x gap dwarfs the steady-state gaps.
5. Prefill flip (target 412, decode guarded): attack route-pack occupancy/tail
   waste (last measured util 61.8%, singleton_tail 656), not post-norm toggles.

## Reproduction

```bash
PROMPT="A teammate sent two local LLM benchmark screenshots with different tok/s numbers for the same model. Explain the likely causes in two short paragraphs, then give one concrete measurement rule."

ZINC_MODEL_ID=gemma4-26b-a4b-q4k-m \
ZINC_METRIC_MODE=decode \
ZINC_PROMPT_MODE=chat \
ZINC_TEST_PROMPT="$PROMPT" \
ZINC_REFERENCE_TEXT="tokens per second" \
ZINC_MAX_TOKENS=96 ZINC_MIN_DECODE_TOKENS=64 \
ZINC_TARGET_TOK_PER_SEC=84 ZINC_STOP_ON_TARGET=0 \
ZINC_BENCHMARK_RUNS=3 ZINC_BENCHMARK_WARMUPS=1 ZINC_BENCHMARK_CONFIRM_RUNS=4 \
ZINC_PROFILE_EVERY=1 ZINC_BUILD_OPTIMIZE=ReleaseFast \
ZINC_CROSS_EFFORT_PROMPT="$PROMPT" ZINC_CROSS_EFFORT_METRIC=prefill \
ZINC_CROSS_EFFORT_PROMPT_MODE=chat ZINC_CROSS_EFFORT_MAX_TOKENS=32 \
ZINC_CROSS_EFFORT_EVERY=3 \
ZINC_HARD_FAMILY_COOLDOWN=1 ZINC_WORKLOAD_RESET_ON_CHANGE=1 \
bun loops/implement_metal.ts \
  --effort-file loops/efforts/MULTI_HOUR_EFFORT_29_METAL_M4_GEMMA26_PUBLIC_GAP.md \
  --agent codex --model gpt-5.5 --cycles 100
```

Prefill flip: `ZINC_METRIC_MODE=prefill ZINC_TARGET_TOK_PER_SEC=412
ZINC_CROSS_EFFORT_METRIC=decode ZINC_CROSS_EFFORT_MAX_TOKENS=96`.
Validate keeps with `bun tools/performance_suite.mjs --target metal --phase all
--models gemma4-26b-a4b-q4k-m --runs 3 --warmup 1 --no-site-write` (drop
`--no-site-write` only to publish).

Keep rules: chat output coherent and contains "tokens per second";
`zig build test` passes; decode improves or evidence names the next decode
bottleneck; prefill guard within −3%; same optimize mode before/after.
