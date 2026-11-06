# MULTI-HOUR EFFORT 30 — CUDA RTX 5090 PREFILL: close the llama.cpp gap

**Goal:** improve ZINC prompt-**prefill** throughput on the RTX 5090 toward / past
llama.cpp, one validated increment per cycle. Decode is already ~85-90% of llama;
prefill is the structural gap.

## Current state (2026-07-06, after the attention-coalescing win landed on main)

The single biggest cheap win of this effort is DONE and on main:
**coalesced warp-per-key prefill attention (`gemma_attention_batched_v2`,
`attention_causal_batched_v2`, default-on `ZINC_ATTN_V2`).** It fixed the naive
uncoalesced pass-1 score loop (256 threads read 256 keys strided by kv_stride →
~32 uncoalesced txns/element). Measured on the 5090:

- **gemma-31b (dense): +53% prefill (552→847 t/s), gap to llama 6.0×→3.9×** ← headline
- gemma-26b (MoE): +7% (experts dilute it)
- qwen: marginal (attention is a small fraction of SSM-heavy qwen prefill)

## Measure-FIRST discipline (this is how the win was found — honor it)

Phase-profile before optimizing. Tools already in the tree:

- `ZINC_PREFILL_PROFILE=1` (qwen path, forward_cuda.zig) → `attn/ssm/ffn` split.
- `ZINC_SSM_PROFILE=1` → SSM `scan` timing.
- For gemma, add throwaway `waitPending`+`nanoTimestamp` phase timers (single
  stream → GPU order unchanged → reliable RELATIVE breakdown; revert, never commit).
  The BT=32 negative below was the ONE guess made without profiling — it was neutral.

## Gemma-26b MoE prefill phase split (T=1042, post-v2): attn 312 / experts 264 / shared 133 / router 23 / combine 24 ms

## Qwen prefill: SSM dominates (qwen35-9b 52% / qwen36-27b 80% / a3b ~65%); the SCAN is cheap (5-9ms/layer) — the cost is the SSM PROJECTION GEMMs (all cuBLAS-eligible).

## Candidate levers (priority = EV × tractability). Pick ONE per cycle, profile-gate it.

1. **int8 MMQ TC GEMM (THE priority — it's the only lever left; hard, multi-cycle).**
   Attacks attention-QKVO + experts + SSM-projections at once. The correct-SASS
   CUBIN fp16 mma path exists + is token-correct but LOSES to cuBLAS (fp16-TC =
   cuBLAS parity ceiling is real). int8 is the only thing that can beat cuBLAS
   (2× TC rate + int8 nibbles direct). DESIGN: keep the Q4_K nibble as int8,
   quantize the activation int8 per-row (Q8_1-style), `mma.sync …s8.s8.s32`;
   Q4_K-asymmetric epilogue = accumulate per-32-subblock s32 (`P=Σ nib·qA`) →
   store to shared → fp32 fold `acc += sA·d·sc·P − sA·dmin·mn·SA[sb]`. THE RISK:
   that per-subblock store-s32-to-shared+rescale is a tax the fp16 path avoids →
   a prior microbench (Effort-26 cycle-8) killed int8 to <1.3×. So: build an
   ISOLATED `dbg_cuda gemm M K T` microbench of the int8 kernel vs `gemm_q4k_tc`
   / cuBLAS at gemma shapes FIRST; if not ≥1.3× ISOLATED, abandon (do NOT wire).
   Only wire into gemmDispatch if the microbench passes. Compile via the standalone
   CUBIN path (nvcc, correct s8 SASS) — NVRTC miscompiles TC on sm_120.
2. **QKV projection fusion** (attention GEMMs, ~200ms): Q/K/V read the SAME b.norm
   input with 3 separate cuBLAS GEMMs → one grouped/concatenated GEMM (dequant 3
   weights into one fp16 buffer, one cublasGemmEx, slice outputs). Watch the
   gemma SWA/global V-variant (some layers have no separate Wv). Modest (+2-5%).
3. **qwen SSM conv1d** (`ssm_conv1d_batched`, F32): grid `conv_channels/64` = low
   block count (160 for 27b, ~1 block/SM). Possible naive-parallelism win, but
   it's a small fraction of the SSM — PROFILE its share first.
4. **int8 MMQ TC GEMM** (LAST resort, hard, uncertain): the only lever left for
   the cuBLAS GEMM wall. Reads Q4_K nibbles as int8, Q8_1 activation, `mma.sync
s8.s8.s32` (2× TC rate). The correct-SASS CUBIN path exists + is integrated
   (ZINC_PREFILL_MMA, output token-correct). BUT the Q4_K-asymmetric per-subblock
   store-rescale EPILOGUE TAX is STRUCTURAL (a prior microbench killed int8 to
   <1.3×). Only attempt with an isolated microbench gate (≥1.3× vs cuBLAS) BEFORE
   wiring. Multi-cycle.

## Dead ends — DO NOT re-litigate (all tested this effort, negative)

- **FLASH-ATTENTION (`gemma_attention_flash`, query-tiled online-softmax)**: DEAD.
  Cycle-5's WIP builds but produces EMPTY output (crashes/deadlocks the GPU — a
  __syncthreads/OOB bug; it hung cycle-5 4.75h). Premise is marginal anyway: K/V
  is small + L2-cached, so v2's coalescing already captured the reuse. Cycle-3
  measured a working flash variant NEGATIVE at T=376. Do NOT rebuild flash.
- **BT=32 MoE expert tiles**: neutral (padding not the bottleneck; weight-dequant-ALU-bound).
- **qwen attention v2**: marginal (attention small in SSM-heavy qwen) — opt-in, don't default-on.
- **fp16 weight cache / ZINC_PREFILL_F16**: warm/serving-only; −24% on cold single prefill.
- **CUBIN fp16 mma (gemm_q4k_mma_lowsmem, Q4_K-direct TC)**: −11% vs cuBLAS. The
  fp16-TC = cuBLAS-parity ceiling is REAL (correct SASS, still loses). fp16 hand
  GEMM is DEAD; only int8 could beat cuBLAS, gated by the epilogue-tax microbench.
- **Prefill CUDA graphs / TC-default micro-opts / m128 / normf16 / FP8**: all prior negatives.

## HARD RULES (override the generic playbook)

- **Pin the 5090**: `export CUDA_VISIBLE_DEVICES=${ZINC_GPU}`
  and `ZINC_GPU=` the same. The 4090 (${ZINC_GPU}) may be used by other loops.
- **Box build dir `~/zinc-harvest`** (a full main checkout, NOT a git repo — rsync
  the WHOLE worktree `./ dest/` single-source, never multi-source+--delete which
  scrambles the tree). Build: `./scripts/zig.sh build -Dbackend=cuda -Dshaders=false -Doptimize=ReleaseFast`.
- **Correctness gate**: `validate_catalog.sh` is UNUSABLE in a non-.git box tree
  (its `zig build cuda-dbg` auto-RUNS the binary → FileNotFound). Gate instead on
  **default-vs-change greedy token match** (`--prompt "<real text>" --raw -n 20`,
  compare the `Output (…)` line): a change that is token-identical to the shipped
  default is as-correct-as-default. For token-tolerance kernels (reduction reorder)
  require the tokens to MATCH on ≥2 real prompts across gemma-26b + gemma-31b.
- **A/B**: interleaved ABBA, ≥4 rounds, discard the cold first round; the box has
  ~±10% boost noise — require a consistent multi-round win. `Prefill complete: … tok/s` is on STDERR (2>&1). Models reload per process (gemma-31b 18GB ~90s).
- **Git**: the main checkout `${ZINC_WORKTREE}` may be owned by a
  parallel loop → `git checkout` can abort. Use `git worktree add` for all branch
  work; commit ONLY your scoped change to a `perf/e30-<target>` branch and push it
  (NEVER main). If a win, append a dated entry here + to `project_effort26_beat_llama` memory.
- Revert + log negatives (they're valuable). Clean box scratch. STOP after one increment.

## CONVERGED (2026-07-08) — effort closed, every prefill lever resolved

**THE WIN (shipped to main `ac8192c4`):** coalesced warp-per-key attention (`ZINC_ATTN_V2`
default-on) → **gemma-31b dense prefill +53% (552→847 t/s, gap 6.0×→3.9× vs llama)**;
gemma-26b MoE +7%. This was the sole cheap win — a genuinely naive uncoalesced softmax.

**THE LAST LEVER, decisively KILLED (int8-MMQ):** built the full Q4_K int8
`mma.sync.m16n8k32.s8.s8.s32` GEMM microbench (`dbg_cuda gemm8`/`mma8`, harness on this
branch's `src/dbg_cuda.zig`). Findings on the 5090 (M=K=4608):

- NVRTC **compiles inline-PTX int8 mma correctly on sm_120** (no CUBIN gamble) and Blackwell
  delivers the **1.9× int8 TC rate** — both feared unknowns settled favorably.
- **But int8/fp16 = 0.92–0.98× (SLOWER), correct to 0.38%** → dense Q4_K prefill is
  **weight-traffic-bound**, so int8's 1.9× _compute_ rate buys **0 wall-clock** (both read the
  same 0.5-byte Q4_K weight). int8-MMQ is DEAD. Reproduce: `dbg_cuda gemm8 4608 4608 512`.

**Every other lever = dead** (see Dead-ends): flash (broken+L2-caches-KV), BT=32, qwen-attn
(marginal), QKV-fuse (hangs), conv1d, Q8_0-dense (weight-traffic ×2), fp16-cache (warm-only),
CUBIN-fp16-mma (−11%), m128/normf16/FP8/graphs. **Prefill is weight-traffic-bound → the only
lever left is LESS weight traffic (lower-bit quant / different format), not a faster kernel.**
Residual awake-only idea: cuBLAS-per-MoE-expert (predicted-neg — experts already grouped-TC fp16).

## Cycle log

(append dated one-liners per cycle: target, verdict, branch)
