# MULTI-HOUR EFFORT 31 — CUDA RTX 5090 DECODE: close the llama.cpp gap

**Goal:** improve ZINC single-token DECODE throughput on the RTX 5090 toward llama.cpp,
one validated increment per cycle. Prefill effort 30 is CONVERGED (+53% attn shipped;
rest = weight-traffic wall). Decode is now the axis.

## Measured 5090 decode gaps (zinc med tok/s vs llama-bench tg160, 2026-07-08)
gemma4-26b MoE 51/157 = **32%** (worst); qwen36-35b-a3b MoE 62/158 = **39%**; gemma4-31b
32/61 = 52%; qwen35-9b 91/157 = 58%; qwen36-27b 42/63 = 66%. **MoE rows are the worst.**

## Root cause (MEASURED, measure-first — honor it before optimizing)
- **CUDA graphs are NEUTRAL on the 5090** (qwen35-9b graph-on vs ZINC_CUDA_GRAPH=0 ~equal;
  graph VERIFIED engaged). Decode is GPU-**compute/latency-bound, NOT launch-bound.** Do
  NOT chase graphs (gemma has none — porting = predicted-neutral, Effort-27 C4 4090 confirms).
- **gemma-26b decode phase split (ZINC_DECODE_PROFILE): embed 0.09 / LAYERS 13.6 (90%) /
  tail_LMhead 1.03 ms/tok.** The vocab=262144 LM head is only 7% (NOT the lever). The layers
  (attention QKVO + MoE expert matvecs) dominate at **~8% memory BW** — at T=1 too little
  parallelism to hide memory latency (~750 tiny kernels/tok × ~18µs GPU exec-latency each).
- ⇒ **the lever is FEWER, BIGGER launches (static kernel FUSION), NOT graphs** (fusion cuts
  the kernel COUNT the graph can't). This is the ONLY viable decode lever found.

## Candidate levers (priority; pick ONE per cycle, token-gate it)
1. **Static kernel fusion on the MoE decode hot path (Effort-27 playbook — the proven
   win-class).** Fuse ADJACENT single-block launches across command boundaries into ONE
   launch (bit-identical, block-count preserved). Effort-27 landed many +3-12% here on the
   4090 (gate+up dual, combine-tail, pre-norm triple, norm-combine, attn-moe-norm). SCAN
   forward_cuda_gemma.zig's decode path (moeFfnBlock / attentionLayer / the tail) for
   REMAINING un-fused adjacencies: two consecutive single-block rms/elementwise launches
   reading the same or chained buffers → one kernel. The 5090 is MORE launch/latency-bound
   than the 4090 (8% BW) so these should help MORE here than where they were first measured.
2. **Attention QKVO decode matvec** — big matvecs (q_dim=8192 for gemma-26b), check the
   dmmv geometry saturates the 170 SMs at T=1 (blocks≥~340). Low-parallelism = low BW.
3. **Expert matvec parallelism** — M=704 expert GEMMs are skinny at T=1 (704 blocks < ideal);
   grouping active experts for more parallelism. (Effort-27 already did dmmv_q4k_experts.)

## Dead ends — DO NOT re-litigate
- **CUDA graphs (dense + MoE)**: NEUTRAL on 5090 (compute/latency-bound, not launch-bound).
- **LM-head tail optimization**: only 7% of gemma-26b decode — not worth it.
- The prefill dead-ends (int8-MMQ, flash, etc.) don't apply to decode but the *wall* does:
  matvec efficiency at T=1 is fundamentally hard (llama's edge is latency-tuned matvecs).

## HARD RULES (override the generic playbook)
- **Pin the 5090**: `export CUDA_VISIBLE_DEVICES=GPU-5126d018-ec86-be8b-1bf5-b5ac323d3350`, ZINC_GPU= same.
- **Box build dir `~/zinc-harvest`** (full main checkout, NOT git — rsync WHOLE worktree `./ dest/`
  single-source). Build `~/zig-0.15.2/zig build -Dbackend=cuda -Dshaders=false -Doptimize=ReleaseFast`.
- **Correctness gate**: default-vs-change greedy token match on gemma-26b AND qwen — `--prompt
  "<real text>" --raw -n 20`, compare the `Output(…)` line (STDERR, 2>&1). Fusion is usually
  BIT-identical (same kernels reordered) → require token-IDENTICAL. A DIFFER = revert.
- **A/B DECODE**: interleaved ABBA ≥4 rounds, drop cold first round; decode is boost-noisy
  (±10%). Parse `Generated N tokens in … — X tok/s`. Use a ramble prompt (repeat a sentence
  ~8×) so the model generates ≥100 tokens (short prompts EOS early). Models reload/process.
- **Git**: main checkout may be owned by a parallel loop → `git checkout` can abort. Use
  `git worktree add` for branch work; commit ONLY your scoped change to `perf/e31-<target>`,
  push it (NEVER main). Win → append dated one-liner here + to [[project_5090_decode_gap]] memory.
- TIMEOUT: wrap every box zinc run in a timeout; a decode kernel bug can hang the GPU.
- CONVERGENCE SELF-STOP: if genuinely converged (every lever shipped/dead, no valid 50-min
  increment) AND a prior cycle already said so, write /tmp/e31_converged (one-line reason) and
  STOP — do not append redundant confirmations. Revert + log negatives. STOP after one increment.

## Cycle log
(append dated one-liners per cycle: target, verdict, branch)
- 2026-07-08 C1 — gemma-26b MoE decode, the two un-merged Effort-27 fusion flags
  (ZINC_MOE_NORM_COMBINE C17 + ZINC_ATTN_MOE_NORM C19; both already implemented &
  env-gated, both 4/4-variant BIT-IDENTICAL output on the 5090). Since they're
  env-gated I A/B'd them with the SHIPPED binary (no code change). ABBA 16-run
  interleaved, gemma-26b -n128 ramble: **NEUTRAL** — warm-round (drop-cold)
  medians A_default=75.09 / B_both=75.02 tok/s (ratio 0.999), drift-corrected
  per-loop mean Δ(B−A)≈−0.03. **VERDICT: NEGATIVE, no code landed.** These fusions
  were +9.8%/+2.9% on the *4090* but are DEAD on the 5090 → cutting single-block
  norm launch-count buys nothing here. **Refutes the effort's "5090 is MORE
  launch-bound → fusions help MORE" premise**; instead CORROBORATES the root-cause
  (5090 decode is compute/latency-bound, NOT launch-bound — same reason graphs are
  neutral). ⇒ the Effort-27 single-block-norm fusion playbook does NOT transfer to
  the 5090. Future cycles: any *new* single-launch norm/elementwise adjacency will
  also be ~neutral; the real 5090 decode lever is matvec efficiency (weight-traffic
  wall), not launch-count. Not declaring convergence yet (cycle 1, no prior said so).
