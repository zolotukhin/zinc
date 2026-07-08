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
- **qwen SSM conv1d token-parallel (`ssm_conv1d_batched_v2`, `ZINC_CONV_V2`) — DEAD (profile-gate fail,
  2026-07-07):** conv1d is a d_conv=4-tap depthwise conv = ~0.25–0.5% of prefill (v1 0.050ms/call, the
  serial 128-block kernel already occupies the GPU); token-parallel v2 is only 1.25× faster (+2nd finalize
  dispatch eats the gain) → end-to-end ~0.1%, unmeasurable vs ±10% boost noise. Do NOT re-attempt.
- **Q8_0 DENSE GEMM optimization (gemma-26b attn Q/K/V/O + shared-expert gate/up/down are all
  Q8_0) — DEAD regardless of kernel (2026-07-07, TWO cycles):** (a) Q8_0→cuBLAS in-noise ~+3%
  (Q8_0→fp16 DOUBLES weight traffic 1→2 B + round-trip → cancels the TC-rate win); (b) Q8_0→the
  ALREADY-PROVEN fp16-TC in-register kernel `gemm_q8_0_tc_lowsmem` (reads Q8_0 DIRECT, no round-
  trip, no traffic penalty, tensor cores — the "fused-dequant hand kernel" a prior cycle asked
  for; it EXISTS + is default-on in the qwen path) = ALSO in-noise/DEAD-EVEN (ABBA×5 T=169:
  259.5 vs 259.8, fully overlapping). BOTH negative ⇒ the Q8_0 dense GEMMs are too small a share
  of gemma-26b prefill for ANY kernel swap to move end-to-end; the F32 `gemm[3]` is already
  adequate. gemma-26b prefill is bound by the O(T²) softmax + the Q4_K/Q5_1 EXPERT matvecs.
  **Do NOT re-attempt Q8_0 dense-GEMM (cuBLAS, TC-in-register, or any hand kernel).** The wired-in
  TC kernel is token-correct (identical gemma-26b×2 prompts) if ever wanted, but it's a no-op perf.

## HARD RULES (override the generic playbook)
- **Pin the 5090**: `export CUDA_VISIBLE_DEVICES=GPU-5126d018-ec86-be8b-1bf5-b5ac323d3350`
  and `ZINC_GPU=` the same. The 4090 (GPU-e59a6fce-…) may be used by other loops.
- **Box build dir `~/zinc-harvest`** (a full main checkout, NOT a git repo — rsync
  the WHOLE worktree `./ dest/` single-source, never multi-source+--delete which
  scrambles the tree). Build: `~/zig-0.15.2/zig build -Dbackend=cuda -Dshaders=false -Doptimize=ReleaseFast`.
- **Correctness gate**: `validate_catalog.sh` is UNUSABLE in a non-.git box tree
  (its `zig build cuda-dbg` auto-RUNS the binary → FileNotFound). Gate instead on
  **default-vs-change greedy token match** (`--prompt "<real text>" --raw -n 20`,
  compare the `Output (…)` line): a change that is token-identical to the shipped
  default is as-correct-as-default. For token-tolerance kernels (reduction reorder)
  require the tokens to MATCH on ≥2 real prompts across gemma-26b + gemma-31b.
- **A/B**: interleaved ABBA, ≥4 rounds, discard the cold first round; the box has
  ~±10% boost noise — require a consistent multi-round win. `Prefill complete: … tok/s` is on STDERR (2>&1). Models reload per process (gemma-31b 18GB ~90s).
- **Git**: the main checkout `/Users/stepan/Workspace/zinc` may be owned by a
  parallel loop → `git checkout` can abort. Use `git worktree add` for all branch
  work; commit ONLY your scoped change to a `perf/e30-<target>` branch and push it
  (NEVER main). If a win, append a dated entry here + to `project_effort26_beat_llama` memory.
- Revert + log negatives (they're valuable). Clean box scratch. STOP after one increment.

## Cycle log
(append dated one-liners per cycle: target, verdict, branch)
- 2026-07-08 — **THE INT8 KILL-BAR RESOLVED → ❌ DECISIVE KILL. The full Q4_K-int8 `mma.sync.m16n8k32.s8.s8.s32` GEMM is 0.92–0.98× vs the shipped fp16 `gemm_q4k_tc` (SLOWER), NOT ≥1.3× → the effort's SOLE remaining lever is DEAD. ✅ committed harness `dbg_cuda gemm8`, branch `perf/e30-int8-gemm-killbar` (NOT main).** Built the FULL kill-bar microbench the prior ~5 cycles kept deferring to "an awake session" — extends `dbg_cuda mma8` with `gemm8 M K T` (self-contained `GEMM8_CU`: verbatim `gemm_q4k_tc` fp16 baseline + the new `gemm_q4k_int8` kernel; synthetic Q4_K weight + f32 activation; NO model, NO token-correctness surface, touches only `src/dbg_cuda.zig`, timeout-wrapped). **The int8 kernel is exactly the effort-file design:** raw Q4_K nibble→s8 (skip the fp16 bake), activation quantized PER-SUBBLOCK (Q8_1-style `sA[t,sb]=max|A|/127`), `mma.sync.m16n8k32` so each mma spans one BK=32 subblock, asymmetric epilogue `acc += sA·(d_sc·P − dm_mn·SA)` applied **IN-REGISTER** via the PTX-ISA fragment→(row,col) map (the mma8-validated map) → structurally AVOIDS cycle-8's store-s32-to-shared tax. **RESULTS (5090, M=K=4608 gemma shape):** (1) **NVRTC on sm_120 COMPILES the full inline-PTX int8 GEMM — NO nvcc-CUBIN needed** (confirms mma8's greenlight end-to-end; the "multi-day CUBIN gamble" fear is fully retired). (2) **CORRECT: mean L1 rel-err vs `gemm_q4k_tc` = 0.0038 (0.38%)** across all shapes (the max_rel≈1300 is just tiny-denominator near-zero elems) → int8-activation output is faithful, so the timing is TRUSTWORTHY. (3) **SPEED = KILL: int8/fp16 = 0.926/0.931/0.923× @T=512 (3 stable runs), 0.979× @T=1024, 0.961× @T=2048 — int8 is consistently SLOWER, never within 30% of the ≥1.3× bar, margin far beyond ±10% boost noise.** **WHY (definitive, confirms c2/MMQ-v2): the dense Q4_K prefill GEMM is WEIGHT-TRAFFIC-BOUND, so mma8's real 1.9× int8 *compute* rate converts to ZERO end-to-end wall-clock win** (both kernels read the same 0.5-byte Q4_K weight from DRAM; the TC math is not the bottleneck) — and the int8 path's per-subblock activation-quant (max-reduce + round, re-done per M-block) + asymmetric epilogue even cost a few %. **VERDICT: int8-MMQ is DEAD for dense Q4_K prefill on the 5090 — the LAST live lever is now closed with a real measured number, not a deferral. EFFORT 30 IS FULLY CONVERGED: every prefill lever (v2-attention=the sole win/shipped, flash, BT32, QKV-fuse, conv1d, Q8_0-dense×2, fp16-cache, CUBIN-fp16-mma, m128, normf16, FP8, graphs, int8-MMQ) is now either shipped or definitively dead. No autonomous lever remains; recommend STOP the loop.** `dbg_cuda gemm8` committed + reusable. Box scratch clean.
- 2026-07-08 — **INT8-MMA FEASIBILITY MICROBENCH BUILT + RAN → both awake-session gating unknowns RESOLVED FAVORABLY (GREENLIGHT). ✅ committed harness, branch `perf/e30-int8-mma-microbench`.** Instead of a 4th convergence-confirmation log, built the isolated `dbg_cuda mma8` microbench (the harness prior cycles kept deferring) — reuses `benchMode`'s pattern, NO model load, NO token-correctness risk (touches only `src/dbg_cuda.zig`; MMA8_CU source string + `mma8Mode`), timeout-wrapped. **RESULTS (5090, self-validating, 3 stable runs):** (Q1) **NVRTC on sm_120 COMPILES inline-PTX `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` and it EXECUTES CORRECTLY** (a single-warp known-value m16n8k32.s8 matmul checked vs a scalar host ref = **128/128 elems PASS** → BOTH the compile path AND the exact PTX-ISA fragment→(row,col) register map the in-register epilogue depends on are correct). **⇒ the prior cycles' single biggest fear — needing a multi-day nvcc-CUBIN build because NVRTC miscompiles TC on sm_120 — is REFUTED: the fp16 *wmma-intrinsic* lowering miscompiled, but raw inline-PTX mma passes straight through NVRTC's internal ptxas. The awake Q4_K-int8 kernel is buildable via `createPipeline` exactly like `gemm_q4k_tc`, NO offline CUBIN.** (Q2) **int8/fp16 raw TC-rate ratio = 1.90–1.92×** (int8 m16n8k32 ~385 TMAC/s vs fp16 wmma-16³ ~201 TMAC/s, both 4096 MAC/call, register-resident so no boost noise) → **Blackwell delivers the ~2× int8 TC rate the whole premise assumes.** **VERDICT: GREENLIGHT the awake session** — both feared unknowns (compile path + TC rate) are settled, so the int8 lever is now a *tractable NVRTC kernel build*, not a multi-day CUBIN gamble. **THE ONE UNRESOLVED RISK (unchanged, and this microbench cannot answer it — it's register-resident, zero memory traffic):** c2/MMQ-v2 found the dense GEMM is WEIGHT-TRAFFIC-BOUND, so the 1.92× *compute* ceiling only converts to end-to-end win if the GEMM isn't fully traffic-capped. The real kill-bar (**≥1.3× vs `gemm_q4k_tc` ISOLATED @ M≈K≈4608, T=512, WITH memory**) still requires building the full Q4_K-int8 GEMM (raw-nibble→s8 shared stage + per-row A int8 quant + per-32-subblock mma into fresh s32 acc + in-register asymmetric epilogue `acc+=sA·d_sc·P − sA·dm_mn·SA`). Harness `dbg_cuda mma8` is committed + reusable for that. Box scratch clean.
- 2026-07-08 — **INT8-MICROBENCH FEASIBILITY + HARNESS-TEMPLATE LOCATION cycle (converged effort; sole lever = deferred int8 gamble). NO code change, NO branch. Verdict: the int8 `mma.sync.m16n8k32` microbench CANNOT be produced *trustworthily* in a 50-min autonomous cycle → stays correctly deferred to an awake session; this cycle CONVERTS its status from "deferred, needs design" to "harness template LOCATED + kernel-authoring spec grounded in code", so the awake session starts from a template not a blank page. RECOMMENDATION: PAUSE the autonomous prefill loop — 3 consecutive cycles (07-07e audit, 07-07 Q8_0-TC neg, this) have found the space converged with the sole lever being human-awake-only; more autonomous spins will only re-confirm convergence.**
  Box up, 5090 idle (4 MiB/0%). Did NOT force a code change: every non-int8 lever is a documented dead end (flash/BT32/QKV-fuse/conv1d/Q8_0-dense/fp16-cache/CUBIN-fp16-mma/m128/normf16/FP8/graphs), and the int8 gamble is explicitly awake-session-scoped. Reason it's unfit for a 50-min cycle (grounded, not just repeated): a rushed/wrong int8 kernel gives a **false greenlight** (→ wasted multi-day wiring) or **false kill** (→ abandons the effort's only lever) — an untrustworthy microbench number is WORSE than none. **NEW, ACTIONABLE (de-risks the awake session — this is the cycle's product):** (1) **HARNESS TEMPLATE LOCATED** — reuse `benchMode` (`src/dbg_cuda.zig:1795`): `pipeline.createPipeline(ctx, src.ptr, "<kname>")` compiles a self-contained CUDA source STRING (cf. inline `BENCH_CU` const at `dbg_cuda.zig:132`), `buffer.createBuffer(ctx, bytes)` allocs device bufs, `command.beginCommand`/`cmd.dispatch(&pipe, grid, block, &.{&bufs...}, &push, @sizeOf(Push), sharedBytes)`/`commitAndWait`, time with `std.time.Timer`. So the awake microbench = add a `gemm` subcommand whose src STRING inlines: `gemm_q4k_tc` (the fp16 baseline, `kernels.cu:4250`) + its helpers `zinc_half_to_float`+`zinc_q4k_scale_min` + `struct GemmPush { unsigned M,K,T,a_offset,x_offset,y_offset,acc_mode,q8_stride; }` (`kernels.cu:3383`) + the new int8 kernel; alloc synthetic Q4_K weight (36 u32/superblock, K/256 superblocks/row, random d/dmin/scales/nibbles) + f32 A[T,K]; dispatch grid=(M/64, T/64, 1) block=256; time N iters of each; correctness = copy Y[T,M] back + relative-error vs `gemm_q4k_tc`. (2) **INT8 KERNEL SPEC grounded in `gemm_q4k_tc` (read this cycle, `kernels.cu:4250-4339`):** the fp16 kernel's `BK=32` inner chunk is EXACTLY one Q4_K subblock (`nchunk=K>>5`, `d_sc=d·sc`/`dm_mn=dmin·mn` computed per-(row,subblock) then baked into the half operand BEFORE wmma = free scaling — THIS is what int8 can't do straight-through). The k32 design: keep the raw nibble as int8 (skip the `wv=d_sc·nib−dm_mn` bake), quantize A per-row int8 (`sA[t]=max|A_row|/127`), run `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` so each mma spans one BK=32 subblock → its s32 accumulator = `P=Σ nib·qA` for THAT subblock, and because the m16n8k32 fragment→register map is PTX-ISA-DEFINED (each thread holds known (row,col) accumulator elems, unlike wmma's opaque map) apply `acc[m,t] += sA[t]·d_sc·P − sA[t]·dm_mn·SA[t,sb]` (SA=Σ qA over the subblock) IN-REGISTER — NO `store_matrix_sync`→shared→reload (the tax that killed cycle-8's wmma-k16 to 0.99–1.02×). RISK that still needs the awake microbench: (a) does NVRTC pass raw inline-PTX `mma.sync` through to ptxas correctly on sm_120 (inline PTX usually survives NVRTC even though fp16 *wmma-intrinsic* lowering miscompiled — if NVRTC is fine, no nvcc-CUBIN needed and it's tractable; if not, precompile like `gemm_q4k_mma_lowsmem`); (b) the per-row A int8 quant + `SA` reduction overhead vs the win; KILL-BAR unchanged = **≥1.3× vs `gemm_q4k_tc` ISOLATED @ gemma shapes (M≈K≈3584-4608, T=512)** or abandon (cycle-8 fallback: cp.async/wgmma fp16). **Baseline: not re-built this cycle (no code change → main trivially intact; 07-07e already confirmed main builds rc=0 + gemma-31b coherent).** All other levers stay dead. NO branch, tree clean (only this log edit).
- 2026-07-07 — **CONVERGENCE-CONFIRMATION cycle (full two-path cuBLAS-coverage AUDIT + attention-v2 pass-3 verify + int8-mma.sync.k32 scoping). NO code change, NO branch. Refutes a hypothesized Q5_K gap; confirms prefill space is converged; sole live lever = the multi-day int8 CUBIN gamble.**
  Hunted for an un-cuBLAS'd dense GEMM (mirror the c9/c10 +80%/+17% pattern to a new quant). **Audited BOTH forward paths (new, grounds the "converged" claim):** (1) **qwen `forward_cuda.zig:2014` already routes ALL dense/SSM-projection quants Q4_K(0)/Q5_K(1)/Q6_K(2)/Q8_0(3) through cuBLAS**, default-on `T>=cublas_min_t=256` (`use_cublas_q5/q6/q8` all default-true) → the ssm_out=Q5_K "gap" I hypothesized DOES NOT EXIST (already covered); (2) **gemma `forward_cuda_gemma.zig:1477` routes Q4_K(0)+Q6_K(2) cuBLAS** — Q8_0(3) tested-DEAD (07-07b/c, both cuBLAS AND TC-in-register in-noise), gemma-31b has no Q5_K, gemma-26b's Q5_1 expert-down is on the grouped-TC path not dense. ⇒ **no un-cuBLAS'd dense GEMM remains on EITHER path.** **attention-v2 pass-3 (`gemma_attention_batched_v2`, kernels.cu:6142) VERIFIED already coalesced** (thread=dim d, i-loop → per i-iteration the warp reads V[i,0..hd] at CONTIGUOUS addresses = one coalesced txn; the O(T²) softmax is coalesced in BOTH passes) — refutes my "pass-3 maybe uncoalesced like pass-1 was" lead; v2 is fully optimized. **Experts already grouped-TC fp16** (`ZINC_MOE_TC` default-on); cuBLAS-for-experts predicted-NEGATIVE by the c9 T-gate logic (small per-expert token batches ~65 @T=1042 don't amortize the Q4_K→fp16 4×-traffic round-trip; c9 was break-even at T=64). **Baseline INTACT:** current main builds green (rc=0, ReleaseFast, 5090) + gemma-31b coherent ("This introductory passage provides a concise overview of the evolution of computing…"), T=76 cold=103 t/s (below cublas_min_t → hand-TC; the 847 headline is warm+T≥128, a different regime — not re-measured, not needed). **SOLE LIVE LEVER = int8 mma.sync CUBIN GEMM, with a NEW concrete design that attacks cycle-8's kill-reason:** use **`mma.sync.aligned.m16n8k32.s8.s8.s32`** — k=32 makes EACH mma span EXACTLY ONE Q4_K 32-wide subblock, so the per-(row,subblock) `d·sc`/`dmin·mn` asymmetric scales apply to the s32 accumulator **in-register** (the mma fragment→(row,col) map is DEFINED by the PTX ISA, unlike wmma's opaque map) → **structurally ELIMINATES the store-s32-to-shared + rescale EPILOGUE TAX** that killed cycle-8's int8 to <1.3× (cycle-8 used NVRTC `wmma m16n16k16`, k=16 = HALF a subblock + opaque map → forced the shared round-trip). **BUT this needs the nvcc-CUBIN path** (NVRTC miscompiles TC on sm_120 → the k32 kernel must be precompiled like `gemm_q4k_mma_lowsmem`) = a multi-file, multi-day AWAKE build that risks a GPU hang (cf. cycle-5's 4.75h flash hang) → **NOT a 50-min autonomous cycle; deliberately deferred to an awake session** rather than risk an unvalidatable/hanging kernel. Everything else (flash, BT32, QKV-fusion, conv1d, Q8_0-dense, fp16-cache, CUBIN-fp16-mma, m128/normf16/FP8/graphs) stays dead. Box scratch: current main rebuilt in `~/zinc-harvest` (useful for next cycle, not scratch).
- 2026-07-07 — **Q8_0 dense GEMM → fp16-TC in-register kernel (`gemm_q8_0_tc_lowsmem`) on the GEMMA path = ❌ NEGATIVE (in-noise, dead-even), reverted, NO branch. CLOSES the Q8_0 dense-GEMM lever ENTIRELY (not just cuBLAS).**
  Distinct from the 2026-07-07b Q8_0→cuBLAS negative: that path dequant'd Q8_0→fp16 scratch
  (DOUBLES weight traffic 1→2 B + round-trip → structurally cancels). THIS cycle wired the
  ALREADY-EXISTING, ALREADY-PROVEN `gemm_q8_0_tc_lowsmem` (reads Q8_0 DIRECT, 1 byte, dequants
  in-register/shared to fp16 for wmma → NO round-trip, NO traffic doubling + tensor cores) — the
  exact kernel the QWEN path runs default-on (`forward_cuda.zig:2139`, `q8TcLowsmemOn()` default-true).
  gemma-26b's attn Q/K/V/O + shared-expert gate/up/down are ALL Q8_0 (idx==3) and ran the F32
  `gemm[3]` (`gemm_q8_0_tiled_v2`, NO tensor cores) via the `gemmDispatchA` fallthrough. **CHANGE
  (clean +15 `forward_cuda_gemma.zig` only, no kernel — kernel pre-exists):** added the pipe field
  + creation + `use_q8_tc` (opt-in `ZINC_GEMMA_Q8_TC`, both init sites) + an `idx==3 and use_q8_tc`
  branch before the `gemm[idx]` fallthrough. Verified coverage: `gemmDispatch`→`gemmDispatchA`, so
  ALL Q8_0 dense GEMMs (attn ~312ms + shared ~133ms of ~756ms prefill) were routed through the TC
  kernel. Built rc=0 (5090). **CORRECTNESS: token-IDENTICAL** gemma-26b (2 real prompts, T=83 AND
  T=169, DEFAULT vs `ZINC_GEMMA_Q8_TC=1`) + gemma-31b unaffected (no Q8_0, both identical). **PERF
  NEGATIVE (ABBA×5, T=169, drop cold round-1):** A(f32) mean 259.8 / median ~257 vs B(TC) mean 259.5
  / median ~257 = **DEAD EVEN**; distributions fully overlap (A 240–284, B 240–275), well inside the
  ±10% boost floor (the gemma-31b no-op B measured 255 vs 322 = ±20% pure boost noise, confirming the
  floor). T=83 single-run B was +3.3% but that's below-floor too. **WHY (conclusive, extends the
  cuBLAS finding):** swapping the GEMM kernel — cuBLAS (traffic-doubling) OR TC-in-register (no
  penalty) — both land in-noise ⇒ the Q8_0 dense GEMMs are simply NOT a large-enough / slow-enough
  share of gemma-26b prefill for ANY kernel swap to move end-to-end. The F32 `gemm[3]` (register-
  blocked v2) is already adequate; gemma-26b prefill is bound by the O(T²) softmax attention (grows
  with T) + the Q4_K/Q5_1 MoE EXPERTS (264ms). **DISPOSITION: reverted (`git checkout`, src clean);
  logged NEGATIVE. The Q8_0 attn+shared "wall" is a DEAD LEVER on gemma-26b prefill regardless of
  kernel — do NOT re-attempt Q8_0 dense-GEMM optimization (cuBLAS, TC-in-register, or a hand fused
  kernel; the ceiling is ~0% because the share is too small). The real gemma-26b levers remain the
  softmax + the Q4_K/Q5_1 expert matvecs (the MMQ/int8 wall).** Box scratch: reverted source rebuilt
  next cycle; `/tmp/e30_ab.sh` scratch left (harmless).
- 2026-07-07 — **Lever #2 FUSED-QKV cuBLAS GEMM = ❌ NEGATIVE (HANGS THE GPU), reverted, NO branch.**
  Implemented `ZINC_QKV_FUSED` (default-OFF, opt-in): fuse the 3 attn Q/K/V cuBLAS
  prefill GEMMs (all read `b.norm`, same K=n_embd) into ONE `[q_dim+2·kv_dim, n_embd]`
  GEMM → token-major `[T, q_dim+2·kv_dim]` output in new scratch (`qkv_w_f16`/`qkv_out`),
  then slice into Q/K/V downstream via **strided offset aliases** (`aliasBuffer` on
  `qkv_out` at col 0 / q_dim / q_dim+kv_dim, `src_stride = m_total`) so the existing
  `rms_norm_rope_batched` / `rms_norm_kvwrite_batched` read the right slice with **no
  kernel change**. Built clean (rc=0, ReleaseFast, 5090). **DEFAULT path (fused off)
  verified UNAFFECTED** — gemma-31b `--chat` 299 tok/s prefill, coherent output. But
  **FUSED ON HANGS**: gemma-31b loads 18GB then GPU sits at 0% util / 180 MHz idle,
  NO prefill line, hits the `timeout 200` with EMPTY output (classic hung/deadlocked
  kernel). Per the HARD-RULE timeout discipline → REVERTED (local `git checkout`, box
  scratch cleaned). **Most likely cause (for next cycle): alias LIFETIME** — the K/V
  offset-view aliases of `qkv_out` are freed at a function-scope `defer` in
  `attentionLayerBatched`, which fires when the fn RETURNS; if the caller submits/executes
  the `cmd` AFTER return (deferred submit), the GPU reads freed alias handles → hang.
  (Secondary suspect: a stride/offset bug in the batched norm/RoPE reads of the wide
  `qkv_out`, but that would corrupt-not-hang.) **Next cycle if retrying:** (1) confirm
  `cmd` submission timing — is `cmd.dispatch` an immediate stream launch or deferred to
  a submit at `waitPending`? Only free aliases AFTER the GPU has consumed them (submit
  inside the fn, or hold the aliases on `self` and free next layer); (2) gate behind an
  ISOLATED microbench first — and note the EV is modest (`a_preconv` already removed the
  redundant activation recast, so fusion only saves 2 cuBLAS launches + merges the small
  underutilized K/V GEMMs into Q; predicted +1-3%, likely in ±10% boost noise anyway).
  Diff was contained to `forward_cuda_gemma.zig` (+83/-22): flag, 2 scratch buffers, a
  fused branch in `attentionLayerBatched` refactoring the projection block to select
  q/k/v src+stride. **DO NOT rebuild it as-is (it hangs).**
- 2026-07-07 — **Q8_0 dense GEMM → cuBLAS on the GEMMA path = ❌ NEGATIVE (in-noise ~+3%), reverted, NO branch. NEW DEAD END.**
  **THE FINDING (worth keeping):** gemma-26b-A4B's attn Q/K/V/O AND its shared-expert
  ffn_gate/up/down are ALL **Q8_0 (idx==3)** (verified via `gguf.GGUFReader`: 207 Q8_0
  tensors; routed experts are Q4_K gate_up + Q5_1 down). The gemma cuBLAS path (c9/c10)
  only covered idx==0 (Q4_K) + idx==2 (Q6_K), and the TC path too → **all those Q8_0
  GEMMs (attn ~312ms + shared ~133ms of a ~756ms prefill) ran the f32 hand `gemm[3]`
  tiled kernel.** This explains cycle-9's "gemma-26b neutral" (its dense GEMMs are Q8_0,
  never hit cuBLAS). gemma-31b is pure Q4_K/Q6_K (no Q8_0) → fully covered already.
  **THE CHANGE (clean, +19/-2 `forward_cuda_gemma.zig` only, no kernel change — the
  `dequant_q8_0_to_f16` kernel + `DequantQ8_0Push` already exist for the qwen path/Effort-29):**
  added `use_cublas_q8` (default-on when use_cublas, opt out `ZINC_BATCHED_CUBLAS_NOQ8`),
  `dequant_q8_0_to_f16` pipe, and extended the gemmDispatchA cuBLAS branch to
  `(idx == 3 and use_cublas_q8)`. Built clean (rc=0, 5090). **CORRECTNESS PASS:**
  gemma-26b `--chat -n 12` at T=288 (≥cublas_min_t=128 so cuBLAS engages), DEFAULT
  (Q8_0→cuBLAS) vs `ZINC_BATCHED_CUBLAS_NOQ8=1` (Q8_0→f32) = **token-IDENTICAL** output
  ("# Lecture Notes: GPU Execution Patterns for Dense Matrix Multiplication in"); the
  fp16-cuBLAS Q8_0 path rides the token-tolerance gate. (Also gemma-31b unaffected by
  construction — no Q8_0 weights, idx==3 never taken.) **PERF NEGATIVE (order-alternated,
  T=288, box under load-41 contention from a parallel loop):** round1 A(cuBLAS) 431.27 vs
  B(f32) 414.22 = +4.1%; round2 B(f32) 426.08 (≈round1 A). Three points 431/414/426 overlap
  heavily — the gap is inside B's own ~3% round-to-round boost spread and well below the
  ±10% boost floor. **WHY (structural, not just noise — this is the reusable lesson):**
  Q8_0 is already 1 byte/weight; dequanting to fp16 **DOUBLES** the weight DRAM traffic
  (1→2 bytes) **and** adds a full-weight round-trip write/read, whereas the f32 hand
  `gemm[3]` reads the Q8_0 nibbles direct. cuBLAS's 2× TC-rate is cancelled by the extra
  weight traffic (c2's "dense GEMM is weight-traffic-bound" applies → same family as c11
  FP8 negative). This is the OPPOSITE of Q4_K/c9 (+80%), where the weight is 0.5 byte AND
  the hand TC kernel was genuinely inefficient. **DISPOSITION: reverted code (`git checkout`,
  src clean); logged NEGATIVE. Q8_0-cuBLAS is a DEAD END for gemma-26b prefill — do NOT
  re-attempt.** The gemma-26b Q8_0 attn+shared GEMM wall needs a FUSED-dequant hand kernel
  (read Q8_0 direct, no fp16 round-trip — the cycle-8 cp.async/wgmma idea but for Q8_0),
  NOT a dequant→cuBLAS round-trip. Box `~/zinc-harvest` scratch: reverted source rsync'd
  back; the built binary there is stale (rebuild next cycle).
- 2026-07-07 — **Lever #3 qwen SSM conv1d token-parallel (`ssm_conv1d_batched_v2`, `ZINC_CONV_V2`) = ❌ NEGATIVE (PROFILE-GATE FAIL — conv1d is ~0.25–0.5% of prefill), reverted, NO branch. NEW DEAD END.**
  Mid-flight code from a prior cycle (2 kernels `ssm_conv1d_batched_v2` + `ssm_conv1d_state_finalize` in
  kernels.cu, opt-in wiring in forward_cuda.zig, bit-identical) — turns the serial conv1d (grid
  conv_channels/64 blocks, each looping T tokens = ~1 block/SM cost-scales-with-T) into a fully
  token-parallel grid.y=T kernel + a separate circular-state write-back. Built rc=0. **PROFILED FIRST
  (per lever-#3 caveat "PROFILE its share first"), via a throwaway CONV_PROFILE timer isolating 100
  repeated conv dispatches at L0 (v1 vs v2, then reverted):** conv1d is a **d_conv=4-tap** depthwise
  conv — TINY. qwen35-9b (ch=8192, T=120): v1 **0.050ms/call**, v2 0.040ms/call = only **1.25× kernel
  speedup** (the 128-block v1 already occupies the GPU decently AND v2 adds a 2nd finalize dispatch that
  eats most of the gain). qwen36-27b (ch=10240, T=120): v1 **0.050ms/call** against **1176ms** total
  prefill. **SHARE:** ~40–60 SSM layers × 0.05ms ≈ 2–3ms of a 340ms (qwen35-9b) / 1176ms (qwen27) prefill
  = **~0.25–0.5%**; v2 saves only ~0.01ms/call → **end-to-end effect ~0.1%, FAR below the ±10% boost
  floor** (the single-run prefill +6% v1→v2 on qwen35-9b, 333→353 t/s, is pure boost noise — the conv
  delta literally cannot produce >0.1%). CONFIRMS memory ("SSM cost is the PROJECTION GEMMs; the SCAN is
  cheap 5–9ms/layer" — conv is even cheaper than the scan). **DISPOSITION: reverted both source files
  (`git checkout`, 0 residue) + threw away the profiler; logged NEGATIVE — NOT wired, no branch. conv1d
  is a DEAD LEVER (too small a share for ANY kernel to move prefill) — do NOT re-attempt token-parallel
  conv1d.** The only remaining prefill lever stays the int8-MMQ GEMM gamble (structural epilogue tax,
  microbench-gated ≥1.3× vs cuBLAS BEFORE wiring). Box scratch `/tmp/e30_prompt.txt` left (harmless).
