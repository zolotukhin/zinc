# Optimization 6: RDNA Prefill Recovery for Qwen3.5/3.6-35B-A3B

## TL;DR  (2026-04-26, after 50 autonomous-loop cycles)

**Effort 6 has moved prefill from 78.11 → 90.24 tok/s (+15.5%) over 50
cycles.** All gains came from **micro-restructures of existing
shaders**, NOT from the multi-week tiled-GEMM port that earlier
cycles attempted.

What landed (5 winning cycles):

| cycle | tok/s | delta | what changed |
|---|---:|---:|---|
| 22 | 80.61 | +2.5 | wire count_experts post-prefill sweep (foundation) |
| 28 | 83.07 | +2.5 | rms_norm_dmmv_f32 NUM_ROWS=2→1 (occupancy) |
| 42 | 86.94 | +3.9 | rms_norm_dmmv_f32 vec4 reads/writes (cache-line) |
| 46 | 87.82 | +0.9 | rms_norm_mul vec4 + per-thread reg cache |
| 50 | 90.24 | +2.4 | ssm_delta_net 8t×8r → 16t×4r (cache-line + occupancy) |

What was tried and reverted (cycles 14-21 + 40):

The 5-step llama.cpp tiled-GEMM port was **fully implemented as
foundationKeeps**: count_experts (cycle 14), mul_mm_q4k (cycle 16),
mul_mmq_q4k (cycle 18), mul_mm_id_q4k (cycle 21). All four
foundations stayed dormant. mul_mm_q4k was wired into the LM head
only (N=1 — the worst case for tiled GEMM, since the BN tile is
wasted) and measured FLAT (78.14 vs 78.55 noise band). The other
three pipelines had **zero callers anywhere**. Cycle 40 audited and
reverted **1470 LOC** of dormant infrastructure. mul_mm_q4k.comp +
count_experts.comp survived as currently-wired infrastructure;
mul_mm_id_q4k.comp and mul_mmq_q4k.comp were deleted.

**The lesson is not "the GEMM port doesn't work."** The lesson is
**porting foundations without wiring them into the actual hot path
banks no tok/s** — and the wire-up is a "high-risk one-cycle
refactor needed for buffer layout" (cycle 40 self-analysis) that
the loop has refused to attempt in a single cycle.

What's left to attempt (concrete, cycle-sized):

1. **Wire mul_mm_q4k into SSM proj prefill** (the deferred cycle-40
   work). SSM proj fires 4 DMMVs × 30 layers × 154 tokens — perfect
   amortization shape, unlike the LM head. Two-phase plan: build
   gather/scatter helper standalone with synthetic [M, K, N] data;
   wire ONE projection (z is the simplest, M=d_inner) and measure SSM
   proj phase delta.
2. **Apply cycle-50 pattern to MoE inner loops.** dmmv_q4k_moe_kpar
   (M=1408, NUM_ROWS=2) and dmmv_q4k_moe_fused_down_acc are
   structurally similar to pre-cycle-50 ssm_delta. The 884ms MoE
   bucket (24% of prefill) has been untouched since cycle 40.
3. **Parallel-scan SSM prefill** — the largest unattacked structural
   lever. Token-recurrent state is currently scanned token-by-token.
4. **Q pre-load in flash_attn.comp** — eliminates head_dim × block_len
   redundant reads in the Q.K dot loop.

## Current measured gap (long-context prefill, 154-token prompt, after cycle 50)

| model                       | ZINC P  | llama P  | ZINC %  |
|-----------------------------|--------:|---------:|--------:|
| Qwen 3.5/3.6 35B A3B Q4_K_XL |  90.24 |   ~180   |    50% |

Latest site benchmark (short prompt, 5 tokens): ZINC 67-69 tok/s vs
llama.cpp 182-184 tok/s. The 154-token loop benchmark is more
favorable to ZINC (longer prefill amortizes setup overhead).

ZINC remains at parity-or-better on dense models (Qwen 3 8B prefill
164% of llama.cpp). The gap is **MoE+SSM only** and proportional to
MoE/SSM share.

## Phase budget snapshot (after cycle 50)

Total prefill ≈ 1.7 s on the flagship 154-token benchmark (90.24 tok/s).

- **MoE: ~884 ms (24% of prefill)** — biggest bucket, untouched
  since cycle 40 reverted the dormant GEMM foundations
- SSM: ssm_delta ~450 ms (post-cycle-50), ssm_proj ~325 ms, ssm_gnorm
  ~50 ms, ssm_out ~80 ms
- Attention: ~340 ms (q.k dot + flash_attn + RoPE)
- topk: ~117 ms (single-WG single-pass; cross-WG parallelism untried)
- norm + LM head + tail: small (dead-tail-skip already shaves these)

**Don't refresh from this snapshot blindly** — re-run with
`ZINC_PREFILL_PROFILE=1` before targeting any sub-bucket; cycle 50's
ssm_delta restructure shifts the budget.

## Why the GEMV-cross-token approach fails  (kept as record)

Earlier cycles tried `dmmv_q4k_moe_batched.comp` (commit `c36bd23`)
as a structural swing. It dispatches **1.7M workgroups** (M ×
n_experts_used × n_tokens) where R9700 caps at ~1024 in flight. The
shader stays in tree as record but should not be wired. The
empirical evidence from 9 wire-in cycles: flat or negative.

| approach | total WGs | per-WG work | WG saturation on R9700 |
|---|---:|---|---|
| ZINC per-token GEMV (current) | 154 × 8 × 704 = 867k | 1 dot | over-saturated, dispatch-bound |
| `dmmv_q4k_moe_batched.comp` (GEMV cross-token) | 1408 × 8 × 154 = 1.7M | 1 dot | even worse |
| llama.cpp `mul_mm` MUL_MAT_ID (tiled GEMM) | 22 × 1 × 128 = 2,816 | 64×64 tile = 4096 dots | balanced |

The structural answer is per-expert grouped GEMM. The architectural
shape is correct; the foundation port was correct. The missing piece
is the **wire-up into prefillBatched** with the gather/scatter
buffer layout.

## The original port plan  (preserved as historical context — cycles 14-21 + 40 below)

The 5-step plan below was attempted and partially landed:

- **Step 1 (mul_mm_q4k)**: PORTED in cycle 16, wired LM-head only,
  measured FLAT, KEPT in tree.
- **Step 2 (mul_mm_id_q4k)**: PORTED in cycle 21 as foundationKeep.
  Stayed dormant. **DELETED in cycle 40 revert.**
- **Step 3 (count_experts)**: PORTED in cycle 14, WIRED in cycle 22
  behind ZINC_COUNT_EXPERTS_PREFILL=1, KEPT in tree.
- **Step 4 (mul_mmq_q4k Q8_1 variant)**: PORTED in cycle 18 as
  foundationKeep. Stayed dormant. **DELETED in cycle 40 revert.**
- **Step 5 (wire pipeline selection in prefillBatched)**: NEVER
  COMPLETED. Cycle 40 self-analysis flagged this as multi-cycle work.

llama.cpp's MoE prefill is a four-piece system. The reference is
unchanged; what changed is the realization that wiring is harder
than porting:

### 1. Tiled Q4_K GEMM (foundation, no MoE yet)

**Reference**:
- `~/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp`
  — the dispatch shape, tile loop, shared-memory cooperative loads.
  Look at the `#ifndef COOPMAT` branch (warp-tiled MAC) since RDNA4
  has no cooperative_matrix support.
- `~/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_funcs.glsl`
  — block-load + dequant helpers.
- `~/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq_funcs.glsl`
  — Q4_K-specific block decode at lines 303-364 (can be reused; it
  matches our existing `dmmv_q4k_moe_kpar.comp` layout exactly).

**Tile params (start with these, then tune)**:
- `BM=64, BN=64, BK=32, WM=64, WN=32, WMITER=2, WNITER=2, TM=4, TN=2`
- `local_size_x=128` (subgroup_size=64 × NUM_WARPS=2)
- LDS budget: `BM*(BK/2+1)*4 + BN*(BK/2+1)*4 = ~5 KiB` per WG, fits

**Output**: `src/shaders/mul_mm_q4k.comp` + pipeline registration in
`src/compute/dmmv.zig`. Test against the existing per-token Q4_K DMMV
on a synthetic [M, K, N] matmul before integrating with prefill.

### 2. MUL_MAT_ID variant (the MoE gather)

**Reference**:
- `~/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_id_funcs.glsl`
  — the `load_row_ids` function. Each WG scans the `data_ids` buffer
  with subgroup ballot (`subgroupBallot`, `subgroupBallotBitCount`)
  to find tokens whose routed expert matches this WG's `expert_idx`,
  then accumulates them into a `row_ids[BN]` shared list. **This is
  the load-balancing mechanism.** Without it, expert-grouped tiles
  waste cycles on padding tokens.

- `mul_mm.comp` lines 144-145: `gl_WorkGroupID.z = expert_idx`,
  early-exit `if (ic*BN >= data_expert_count[expert_idx])`.

**Output**: same file as #1 with `#ifdef MUL_MAT_ID` guards added,
plus an early-exit using `data_expert_count`. Most of the GEMM body
is unchanged.

### 3. count_experts.comp (trivial helper)

**Reference**:
- `~/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/count_experts.comp`
  — 50 lines, do not modify. Reads the per-token expert_id buffer,
  outputs a `[n_experts]` count buffer that the GEMM uses for
  early-exit. Port directly.

**Output**: `src/shaders/count_experts.comp` + pipeline + dispatch
helper.

### 4. mul_mmq variant (Q8_1-quantized activation)

**Reference**:
- `~/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp`
  — same tiled GEMM as `mul_mm` but the activation is pre-quantized to
  Q8_1 by `quantize_q8_1.comp` (which we already have). The inner-loop
  dot uses `dotPacked4x8EXT` (`GL_EXT_integer_dot_product`).
  R9700 supports this — `vulkaninfo --summary | grep
  integerDotProduct4x8BitPackedSignedAccelerated` returns true.
- `mul_mmq_funcs.glsl` Q4_K branch (lines 303-364) is the integer-dot
  version of the dequant + scale we already do in f32.

**Output**: `src/shaders/mul_mmq_q4k.comp`. Adds another ~2× on top
of #1-3 because activation bandwidth drops 4× (f32 → Q8_1) and the
inner dot uses hardware integer-dot.

### 5. Pipeline selection in `prefillBatched`

After #1-4 land:
- For dense FFN (Gemma 4 31B): use `mul_mm_q4k` with N=N_tokens.
- For MoE FFN (Qwen 3.5/3.6, Gemma 4 26B, GPT-OSS): use
  `mul_mm_id_q4k` with `data_ids` from the existing
  `routing_capture_buf` (forward.zig:885) and `data_expert_count`
  from #3.
- Threshold: switch from per-token GEMV to GEMM when
  `n_tokens > 8` (matches llama.cpp's `mul_mat_vec_max_cols = 8`
  in `ggml-vulkan.cpp` line 267).

Validate via `ZINC_BATCHED_PREFILL=validate` against the per-token
reference. Q8_1 quantization adds ~2^-7 noise per element so the
mmq path validates against `tol=1e-2` instead of `1e-3`; the f32
mul_mm path should match `1e-3` cleanly.

## What not to attempt  (proven dead-ends)

These are the 11-cycle Effort 6 negative results. Do not re-spend
cycles on them:

- **GEMV cross-token batching** (the `dmmv_q4k_moe_batched.comp`
  pipeline). Empirically flat or negative across 9 wire-in cycles.
  Architecturally wrong for RDNA4 at this token-count regime.
  Pipeline + helper stay in tree as record but should not be wired.
- **Three-way RMS+K+V DMMV fusion** (cycle 14 of Effort 10, cycle 11
  of Effort 6). Register pressure collapses occupancy.
- **Wide NUM_ROWS variants on the MoE kpar shader** (cycle 12 of
  Effort 10). Underutilizes occupancy at MoE expert M=1408.
- **Triple-fused MoE swiglu+down+weighted-acc** (cycle 9 of Effort 6).
  Regressed — the existing path's overlap is already good.
- **Barrier scoping computeBarrier→computeBufferBarrier** (cycles 1,
  16, 18 of Effort 10). RADV doesn't differentiate the access masks
  via PipelineBarrier1; would need PipelineBarrier2 to even test
  the hypothesis.

## Existing infrastructure to reuse

- `routing_capture_buf` (`forward.zig:885`, gated `ZINC_CAPTURE_ROUTING=1`).
  Shape `[max_tokens × n_layers × (2 × n_experts_used × u32)]` storing
  `{expert_id, weight_bits}` per (token, slot). Already populated
  during per-token MoE; matches what `mul_mm_id_q4k` needs as
  `data_ids` input.
- `quantize_q8_1.comp` + `pipeline_quantize_q8_1` (commit `5103042`-ish
  era). Activation pre-quantizer; reused as-is by step 4.
- `dmmv_q4k_q8_1.comp` (commit `27f0c76`). Already proves the Q4_K
  integer-dot decode path works correctly on R9700 at GEMV; the
  per-block dequant translates directly to the GEMM inner loop.
- `dmmv_q4k_moe_kpar.comp` Q4_K block decode (lines 100-160). Same
  bit layout as `mul_mmq_funcs.glsl`'s Q4_K branch.

## Validation

Every change must pass `ZINC_BATCHED_PREFILL=validate` on every MoE
catalog model (Qwen 3.5 35B, Qwen 3.6 35B, Gemma 4 26B, GPT-OSS 20B).
Greedy-output check on a 50-token generation must match the per-token
reference for the first 10 tokens minimum.

## Expected outcome

If steps 1-4 ship correctly, Qwen 3.6 35B-A3B prefill on R9700 should
move from **78 → 150-200 tok/s** (closing 40% → 80-100% of
llama.cpp). Gemma 4 26B MoE from **41 → 100-130 tok/s**. GPT-OSS
from **95 → 100-130 tok/s** (smaller relative gain since it's
already at 79%).

## Original plan (preserved below)

## Current State (2026-04-19)

Target device: RDNA4 benchmark node.

Target model: `qwen35-35b-a3b-q4k-xl`

The latest checked-in benchmark artifact in `site/src/data/zinc-performance.json` shows a split story for the flagship model:

- short decode is respectable:
  - ZINC decode median: `73.07 tok/s`
  - llama.cpp decode median: `95.19 tok/s`
- long-context user latency is still badly broken:
  - scenario: `context-long`
  - prompt mode: `raw`
  - prompt tokens: `154`
  - ZINC decode median: `81.11 tok/s`
  - llama.cpp decode median: `111.61 tok/s`
  - ZINC total latency median: `13002.32 ms`
  - llama.cpp total latency median: `231.68 ms`

The site artifact also exposes a benchmark observability gap:

- ZINC `prefill_tps` for this RDNA `context-long` case is currently `null`
- llama.cpp `prefill_tps` for the same case is populated

So the problem is not just "prefill is slow". It is:

1. prefill or TTFT on the flagship RDNA path is still catastrophically worse than baseline
2. our published artifact cannot yet report the ZINC prefill number cleanly for the exact workload we care about

Decode is not the limiting factor for this scenario. The long wait is dominated by prompt ingestion and all of the setup work before the first useful answer token.

## Goal

Turn RDNA prefill on the 35B flagship from a blind spot into an optimization loop with hard numbers, then recover a large fraction of the missing throughput without regressing coherence.

Practical milestones for the RDNA `context-long` workload:

- Phase 0: make ZINC prefill telemetry parse cleanly and appear in the site artifact
- Phase 1: reduce median total latency well below `13.0 s`
- Phase 2: exceed `150 tok/s` prefill
- Phase 3: exceed `300 tok/s` prefill
- Stretch target: close enough to llama.cpp that the remaining gap is attributable to a short list of named bottlenecks rather than "prefill is broken"

The first real win is not a tiny throughput bump. It is restoring trustworthy prefill measurement on the exact flagship workload, then using that metric to drive the loop.

## Benchmark Contract

This effort is scored on `prefill tok/s`, not decode tok/s.

Primary benchmark shape:

- device: RDNA node
- model: `qwen35-35b-a3b-q4k-xl`
- prompt: the same long-context reference packet used by `loops/optimize_perf.ts`
- decode cap: `-n 8`
- build mode: `zig build -Doptimize=ReleaseFast`
- Vulkan env: `RADV_PERFTEST=coop_matrix`
- node state: no competing `zinc`, `llama-server`, or `llama-cli` GPU users

Keep rules:

1. Do not keep a change unless the loop can parse a valid ZINC prefill number for the benchmark run.
2. Do not keep a change that improves prefill but breaks coherence on any of the required RDNA models.
3. Do not accept wins measured only on microbenches if the flagship long-context benchmark stays flat.
4. Track total latency alongside prefill tok/s, because TTFT is the user-visible outcome we are trying to improve.

Supporting measurements that should accompany stalled cycles:

- per-phase prefill timing on the RDNA path (already wired; gate with `ZINC_PREFILL_PROFILE=1`)
- hot-kernel or per-layer timing for the prompt path
- llama.cpp reference re-check when a milestone lands

## Lessons from Prior Runs

This section is not history for its own sake. It is the set of facts the next cycle must start from so we stop relitigating settled questions. The cycle counter is being reset to 1 for the next run; references below are written in terms of *what changed* rather than which numbered cycle did it, because the prior numbering is no longer authoritative.

### What has actually moved the number

Only a handful of changes produced a real best-checkpoint improvement. All share the same shape: restructure *when* work happens, not *how* a single dispatch is tuned.

- **Dead-tail skip (+~3%):** skip `final_norm` + `LM_head` and the last-layer FFN/MoE body for non-terminal prefill tokens. Dead work elimination.
- **Double-buffered prefill pipeline (+~7%):** ping-pong between two command buffers + paired embedding staging. The CPU records token N+1 while the GPU runs token N. Overlap, not per-dispatch tuning.
- **Last-layer attention dead-tail extension (+~2%):** extend the dead-tail skip into the full last-layer attention block (Q/gate DMMV, Q-norm, Q-RoPE, flash_attn, sigmoid, O-proj, residual) for non-terminal prefill tokens.
- **K-parallel non-MoE Q5_K shader (+~3%):** wave64 inner-loop parallelism on a single shader; cumulative direction for Step 8 below.

Best checkpoint as of cycle 50 (2026-04-26): **`90.24 tok/s`** prefill. Target: `150 tok/s` (Phase 2), `300 tok/s` (Phase 3). At 60% of Phase 2 — the remaining gap is now a mix of (a) MoE bucket attack which is structural, (b) the deferred mul_mm_q4k SSM-proj wire-up, and (c) extending the cycle-42/46/50 micro-restructure pattern to remaining hot kernels.

### What has been proven flat on RDNA4 — do not try again without new evidence

The following category has been attempted many times (cumulative movement below 0.25 tok/s). On RDNA4 + RADV, narrowing a single back-to-back `computeBarrier` to a buffer-scoped barrier does not unlock a measurable win.

- Single-buffer or multi-buffer `computeBarrier` narrowings between successive compute dispatches.

Stop proposing cosmetic variations of this pattern. If a future cycle attacks barriers, it must either (a) remove a barrier outright by restructuring what reads what, or (b) come with a micro-benchmark showing that a specific RADV path responds differently.

### NUM_COLS=1 LM-head specialization — also flat

A prior attempt added a compile-time `NUM_COLS=1` specialized variant of `dmmv_q4k_batch.comp` (`SPEC_NUM_COLS` GLSL specialization constant) and routed only the LM head (M > 65536) through it. Net effect: zero. The LM head fires **once per prefill** (and is already skipped for non-terminal prompt tokens by the dead-tail change), so any speedup on that single dispatch is far below the 154-token denominator.

The lesson: **NUM_COLS specialization only matters for dispatches that fire many times per prefill** — i.e. the SSM proj and MoE gate/up/down sites. Routing the LM head through a specialized variant proves nothing.

### Pair-dispatch (num_cols=2) via the existing batch shaders is net-negative on RDNA4

Earlier attempts landed flag-gated pair-batch infrastructure across SSM proj + MoE router + layer-0 peek-ahead. Two measurements were taken:

- Pair-batch with col 0 duplicated to col 1: −0.12 tok/s, SSM proj +23 ms.
- Pair-batch after rewriting `dmmv_q8_0_batch.comp` with wave64 parallelism (2 rows/WG × 64 threads + subgroupAdd) for the cross-token case: still −0.8 tok/s with flag on.

Root cause: even with the shader parallelism partially fixed, the per-layer "stage norm into col0/col1, split 4 outputs back, barrier between" chain costs more than the weight-read amortization saves. Back-to-back single-column DMMVs win via the L2 weight cache for the small-N regime.

**Do not propose another num_cols=2 variant on top of the existing `dmmv_q8_0_batch` / `dmmv_q4k_batch` shaders.** The fix is architectural, not incremental.

The deeper reason our pair-batch attempts lost: the existing `dmmv_q4k_batch.comp` uses **one wave64 thread per output row** (`local_size_x=64`, but the row dot product runs serially inside that single thread, not as a 64-thread reduction). llama.cpp's `mul_mat_vec_base.glsl` does the opposite — `K_PER_ITER=8` quantized elements per thread, multiple threads per row, `subgroupAdd()` reduction across the wave to compute the dot product. Until our DMMV does the same, multi-column batching just multiplies an already-bandwidth-starved single-threaded dot product.

The architectural fixes in priority order — see Steps 8/9/10/11/13 below for concrete plans:

- Step 8: convert single-column DMMV to wave64 K-parallel (subgroupAdd reduction). Precondition for everything else.
- Step 9: NUM_COLS=2..8 compile-time specialized DMMV family on top of Step 8.
- Step 10: mul_mmq Q8_1 activation quantization (2-4× over FP32 dot path on RDNA).
- Step 11: vllm-style grouped MoE GEMM (token permutation + per-expert grouped matmul).
- Step 13: cooperative-matrix mul_mm for prefill projections that exceed DMMV's sweet spot.

### 3-deep prefill pipeline is flat

A prior attempt extended the double-buffered prefill pipeline from 2-deep to 3-deep and measured no change. Submit/wait is already saturated at 2-deep on the 154-token workload; the record+submit+wait gap is not the dominant cost. Do not re-attempt 4-deep without first proving that the CPU record time has materially grown (e.g. from expert-grouping bookkeeping in Step 11).

### What has been repeatedly attempted and rejected as pointless in isolation

- **"Add more phase profiling" without a downstream change in the same cycle.** Earlier attempts added phase/dispatch counters and were reverted because they did not touch tok/s. The profile line *already* exists behind `ZINC_PREFILL_PROFILE=1`; a cycle should turn that flag on, read the output, then act.
- **"Restructure where prompt embedding dequant happens."** Several attempts explored CPU / staging / interleaved variants of the same idea. The current checked-in design (upfront bulk dequant into host-mapped staging) is the accepted equilibrium. Do not resubmit interleaving or async-thread variants without a new bottleneck proof.

### Current phase budget snapshot (refresh before acting)

Numbers below are for the flagship long-context benchmark. They are a snapshot; the loop should re-run with `ZINC_PREFILL_PROFILE=1` before starting a new structural attack so the agent targets the real largest bucket instead of a stale one.

**Stale snapshot (first run, ~28 tok/s):**
- attention: ~0.7 s
- MoE (router + topk + gate_up + swiglu + down + weighted_acc): ~1.6 s
- shared expert: small, typically <0.1 s
- SSM (proj + conv + delta + gnorm + out): ~1.8 s
  - ssm_proj alone: ~1.3 s

**Cycle 50 snapshot (90.24 tok/s, total prefill ≈ 1.7 s):**
- MoE: **~884 ms (24%)** — biggest bucket; untouched since cycle 40
- ssm_delta: ~450 ms (post-cycle-50 — was ~498 ms before)
- attention: ~340 ms
- ssm_proj: ~325 ms
- topk: ~117 ms
- ssm_out + ssm_gnorm: ~130 ms combined

The MoE bucket has flipped to the largest unattacked surface. ssm_proj has been restructured (kpar shaders are the active path); the deferred work is wiring mul_mm_q4k for SSM proj (cycle 40's flagged refactor).

## Dormant Infrastructure Already In The Tree

These code paths were added in earlier cycles and compile into the binary, but have **zero callers in the prefill hot path**. Wiring them in is lower risk than inventing new infrastructure, and every cycle that tries a fresh micro-optimization without wiring them first is leaving free throughput on the floor.

- `DmmvDispatch.recordBatchDispatch` / `recordBatchDispatchPush` in `src/compute/dmmv.zig` — multi-column DMMV entry point supporting `num_cols > 1`. **Exactly one caller** (the LM head fallback at `forward.zig:6103`) and only with `num_cols=1`. The multi-column path has never been exercised against prompt tokens.
- `DmmvDispatch.pipeline_q4k_batch` / `pipeline_q8_0_batch` — loaded pipelines that are only invoked today with `num_cols = 1` and only for the large LM head. Prior pair-batch attempts (see Lessons section above) showed that wiring this with `num_cols=2` on top of the current 1-thread-per-row shader is net-negative; it only becomes useful after Step 8 lands wave64 K-parallel inner loops.
- `DmmvDispatch.recordCoopMatmul` + `pipeline_coop_matmul` (helper at `dmmv.zig:509`) — F16×F16→F32 cooperative-matrix matmul, 16×16 tiles, wave64. **Zero callers anywhere.** Important caveat: as written this consumes only **F16 weights**, but every prefill projection in Qwen3.5-35B is Q4_K/Q5_K/Q6_K/Q8_0. Wiring requires either an upfront f16 dequant pass into a scratch buffer (acceptable only when the dequant cost amortizes across many prompt tokens, e.g. SSM proj × 154) **or** porting llama.cpp's `mul_mm.comp` pattern that dequants quantized tiles into shared memory inside the matmul loop (Step 13).
- `CommandBuffer.computeBuffersBarrier` — multi-buffer barrier helper. Useful when a cycle structurally removes a global barrier (not when it cosmetically narrows one).

What is **not** in the tree, despite being mentioned in earlier commits or docs:

- ~~"device-side per-(token, layer) MoE routing capture" buffer not in tree~~ → **OUT OF DATE**: `routing_capture_buf` IS in tree (forward.zig:885, gated `ZINC_CAPTURE_ROUTING=1`). Useful as `data_ids` input to a future MUL_MAT_ID GEMM port.
- ~~No `quantize_q8_1.comp` shader, no Q8_1 buffer plumbing, no `mul_mmq` pipeline~~ → **OUT OF DATE**: `quantize_q8_1.comp` + `pipeline_quantize_q8_1` ARE in tree. `mul_mmq_q4k.comp` was ported (cycle 18), reverted as dormant (cycle 40), deleted. Re-port + wire is the deferred work, not "Step 10 has to add all three."

Any cycle whose self-analysis proposes "batch X across prompt tokens" and does not reference the existing `recordBatchDispatch` / `recordCoopMatmul` helpers is proposing unnecessary new infrastructure. Check for the existing helper first.

## Big Bets — the path to 2× and beyond

The existing Steps 1–7 covered telemetry, dormant infrastructure wiring, and incremental batching. The first run plateaued at **`28.07 tok/s`**. The second-+third-run loop (cycles 1-50) restructured single-shader hot paths and reached **`90.24 tok/s`** at cycle 50 — 3.2× over the first-run plateau. The remaining gap to llama.cpp's ~180 tok/s is split:

- **Algorithmic ports** still pending: parallel-scan SSM (Step 11-equivalent), grouped MoE GEMM wire-up (Step 11 in this section's nomenclature). These are multi-cycle investments.
- **Micro-restructures** of the cycle-42/46/50 winning pattern still applicable to MoE inner loops (dmmv_q4k_moe_kpar, dmmv_q4k_moe_fused_down_acc) — single-cycle wins.

Each of the bets below is a multi-cycle investment with a concrete shape, a measured reference implementation, and an estimated upside derived from the reference codebase. **NOTE (cycle 50):** Steps 8/10's foundations were ported and reverted in cycles 14-21+40 — see TL;DR. Step 11 (grouped MoE) remains the highest-leverage MoE bet and was never wired.

| Bet | Step | Status (cycle 50) | Reference | Est. upside | Risk |
|-----|------|-------------------|-----------|------------|------|
| Wave64 K-parallel single-column DMMV | 8 | **DONE** — kpar shaders in tree (`dmmv_q4k_batch_kpar`, `dmmv_q5k_moe_kpar`, `dmmv_q6k_batch_kpar`, `dmmv_q4k_moe_kpar`); subgroupAdd-reduction pattern is the canonical inner loop. | llama.cpp `mul_mat_vec_base.glsl` | (already harvested) | — |
| NUM_COLS=2..8 specialized DMMV family | 9 | **TRIED PIECEMEAL, FLAT** — pair-batch via existing batch shaders measured -0.12 to -0.8 tok/s across multiple cycles. NUM_COLS=2..8 on the kpar path is untried but requires different shader plumbing than what exists. | llama.cpp `mul_mat_vec_max_cols=8` | +40–80% on prompt-loop DMMVs (untested at scale) | Low after kpar |
| `quantize_q8_1` + `mul_mmq` integer dot product path | 10 | **REVERTED** — `mul_mmq_q4k.comp` ported in cycle 18 as foundationKeep, stayed dormant, deleted in cycle 40. `quantize_q8_1.comp` survives in tree. Re-doing the same way is a known dead-end; the missing piece is the wire-up. | llama.cpp `mul_mmq.comp` + `quantize_q8_1.comp` | 2–4× on the largest Q4_K/Q5_K prefill DMMVs | Medium |
| Grouped-MoE GEMM (token permute → per-expert grouped matmul → unpermute) | 11 | **NEVER WIRED** — `mul_mm_id_q4k.comp` ported in cycle 21 as foundationKeep, stayed dormant, **DELETED** in cycle 40. The MoE bucket (884ms = 24% of prefill) has been untouched since cycle 40. Highest-leverage MoE bet remaining. | vllm `fused_moe` + llama.cpp `mul_mat_id` | 5–10× on the MoE phase; biggest single MoE win available | Medium (correctness gates required) |
| RDNA shared-memory occupancy limiter on flash_attn + heavy DMMV | 12 | **UNTRIED** | llama.cpp `ggml-vulkan.cpp:2990` | +5–15% on attention; cleans up RDNA cache-thrash | Low (LDS allocation tweak) |
| Coop-matrix `mul_mm` for projections where N >> 8 | 13 | **N/A** — RDNA4 has no `cooperative_matrix` extension; the warp-tiled `mul_mm.comp` (Step 1 of the inner port plan) is what we'd port. Already attempted as `mul_mm_q4k.comp` (cycle 16, kept LM-head-only, FLAT). | llama.cpp `mul_mm.comp` + `mul_mm_funcs.glsl` | (subsumed by Step 1 wire-up of `mul_mm_q4k`) | Medium-high |
| **NEW** Cycle-42/46/50 micro-restructure pattern | 14 | **PARTIAL** — landed +9.4 tok/s across 3 cycles. Targets: dmmv_q4k_moe_kpar, dmmv_q4k_moe_fused_down_acc, moe_weighted_acc. | (in-tree: ssm_delta_net.comp post-cycle-50, rms_norm_dmmv_f32.comp post-cycle-42) | +1–3% per kernel, compounding | Low (well-validated pattern) |

### How these bets compose

These are not parallel bets. They have a deliberate dependency order:

1. **Step 8 is the foundation.** Today's `dmmv_q4k_batch.comp` is single-threaded per row, which is why pair-batching and NUM_COLS=1 specialization both showed flat. Without 64-thread-per-row reductions, every column we add multiplies a starved dot product. Step 8 fixes this on the single-column path first, validates it on flagship coherence, then steps 9/10 stack on top.
2. **Step 9 is a small generalization of Step 8.** Once the inner loop is wave64-parallel, baking `NUM_COLS` into a specialization constant + unrolling the column loop costs ~50 lines of GLSL per variant. Then we wire SSM proj, MoE router, and MoE gate/up/down to call NUM_COLS=2..8 across multiple prompt tokens (this **re-exercises the existing `recordBatchDispatch` plumbing**, not a new helper).
3. **Step 10 is the highest-leverage independent bet.** mul_mmq does not depend on Step 8/9 — it can ship in parallel. The hard part is building the `quantize_q8_1` activation step and a Q8_1-input variant of each weight-quant DMMV/matmul. llama.cpp uses a single Q8_1 quant pass per prefill chunk and reuses the result across every projection in that layer.
4. **Step 11 (grouped MoE) is independent of Steps 8–10.** It restructures the per-prompt-token MoE loop into a per-expert grouped GEMM. Today `forward.zig:4608+` already runs all routed experts in parallel within a single token (`Y=n_used` workgroups). Step 11 inverts this: gather all 154 prompt tokens' routing decisions, group tokens by expert, run **one larger GEMM per active expert** covering its cohort of tokens. With 128 experts and top-8 routing, the average expert receives `154 × 8 / 128 ≈ 9.6` tokens — perfect arithmetic-intensity bump.
5. **Steps 12 and 13 are layered wins.** The RDNA occupancy limiter is a 5-line change (allocate dummy LDS) backed by llama.cpp commit history. Coop-matrix mul_mm is the largest port and should be attacked last when DMMV-side wins are exhausted.

### What we are explicitly NOT going to do under these bets

- **Do not ship NUM_COLS≥2 with the current 1-thread-per-row shader.** Earlier pair-batch attempts have already proved this loses. NUM_COLS specialization waits for Step 8.
- **Do not wire `recordCoopMatmul` against quantized weights without a dequant strategy.** F16-only consumption means Step 13 has to land a dequant-into-shared-memory tile loop *or* an upfront f16 scratch buffer (and amortize the dequant cost across many prompt tokens to be net-positive).
- **Do not build the grouped-MoE permute on the CPU.** llama.cpp's `mul_mat_id` does the gather inside the matmul kernel via `data_ids[]`; vllm does it on the GPU via a permute kernel. CPU permute will be drowned by submit/wait cost on a 128-expert × 154-token workload.

## Working Hypotheses

Treat these as candidates to prove or kill with measurement:

1. The Qwen35 prompt path is still processing tokens too serially, causing repeated weight reads and excessive submission overhead.
2. MoE prompt work is falling back to a decode-style schedule instead of doing any meaningful token grouping or batching.
3. Attention/KV work during prefill still pays per-token barriers or command submission costs that should only exist in decode.
4. Host-side logging and parsing are not exposing prefill timing correctly, so site and loop artifacts lose the metric even when the runtime emits useful data.
5. Some layers already support batched prompt work, but the hot path is forced through narrower interfaces that erase those wins.

The goal of the effort is to turn these from guesses into measured facts, then remove the biggest one first.

## Execution Order

### Step 1: Repair the benchmark and telemetry contract first

Primary files:

- `loops/optimize_perf.ts`
- `tools/performance_suite.mjs`
- `tools/print_test_summary.ts`
- `src/main.zig`
- `src/compute/forward.zig`

Tasks:

- verify the RDNA benchmark path emits a parseable prefill metric on the flagship long-context run
- verify the site data pipeline preserves that metric instead of recording `null`
- make the optimization loop reject runs where ZINC prefill is missing for this effort
- keep the benchmark prompt, prompt mode, and token cap aligned across the loop and the published site workload

Done when:

- the site artifact shows a real ZINC `prefill_tps` value for RDNA `qwen35-35b-a3b-q4k-xl` `context-long`
- the loop uses that same metric as the primary keep/reject signal

Do not start speculative kernel rewrites before this works.

### Step 2: Instrument prefill phase timing on RDNA

Primary files:

- `src/compute/forward.zig`
- `src/compute/attention.zig`
- `src/compute/dmmv.zig`
- `src/vulkan/command.zig`

The per-phase timing is **already implemented** and wired behind `ZINC_PREFILL_PROFILE=1`. It prints three log lines per prefill: a per-token CPU summary, a per-token GPU phase summary, and per-phase totals for MoE and SSM sub-buckets.

What is left here:

- the loop runs with `ZINC_PREFILL_PROFILE=1` on baseline and after every accepted change so the agent prompt always carries a current phase budget. The loop already does this; do not re-add telemetry code.
- if the baseline for a new run does not produce a parseable `Prefill GPU phases` line, fix the parser or the log line, do not change the runtime.

### Step 2.5: SUPERSEDED — see Step 8/9

This step previously proposed wiring `recordBatchDispatch(num_cols=2)` directly into the SSM proj hot path on top of the existing batch shader. Prior attempts proved this is net-negative on RDNA4 because the existing batch shader uses one wave64 thread per output row (see Lessons section). The replacement plan is in Step 8 (wave64 K-parallel inner loop, the precondition) and Step 9 (compile-time NUM_COLS specialization on top of Step 8).

### Step 3: Prove which parts of prefill are still token-serial

Primary files:

- `src/compute/forward.zig`
- `src/compute/graph.zig`
- `src/compute/dmmv.zig`
- `src/compute/elementwise.zig`
- `src/compute/attention.zig`

Questions to answer:

- is prompt ingestion still traversing the decode loop token-by-token in the hot sections? (Yes — `prefillBatch` iterates `decodeStep` per token. Confirmed.)
- which ops already support `n_tokens > 1` and which ones collapse back to one token per dispatch? (`rms_norm_mul` claims support; almost everything else does not.)
- where are we rereading large weight tensors once per prompt token? (SSM proj reads 4 weights × 30 layers × 154 tokens. MoE gate_up/down reads per-expert per-token.)
- where are command buffers or barriers being rebuilt per token when they could be amortized?

Record the answer in the effort log or commit message after the measurement cycle. Future agents should not have to rediscover this.

### Step 4: Take the safest batch wins first — measurable, not cosmetic

Start with improvements that reduce prompt overhead without changing model semantics aggressively.

Allowed targets (each must either change the number of dispatches or the number of weight reads per prefill — cosmetic barrier scope changes are out of scope for this step):

- batched embedding upload or staging reuse where a single dispatch would replace multiple
- batched RMS norm and elementwise passes that already understand `n_tokens`
- projection paths where one weight read can serve multiple prompt columns (hook into Step 9 once Step 8 has landed)
- fewer per-token command submissions, ideally with the double-buffered pipeline-prefill path extended to 3+ deep (only after proving the CPU record cost has materially grown — earlier 3-deep attempts were flat)
- fewer redundant barriers across prompt tokens — only accept if the barrier is actually *removed*, not just narrowed in scope

Acceptance rule:

- measurable prefill gain on the flagship benchmark (>= the usual threshold versus the best checkpoint)
- coherent output still passes across all RDNA models

If these steps do not move the number, that is evidence. Use it to justify deeper MoE work, not as a reason to keep random churn.

### Step 5: SUPERSEDED — see Step 11

This step previously proposed three microsteps (5a/5b/5c) for batching MoE work across prompt tokens, starting with NUM_COLS=2 on the router DMMV. The router-batching microstep is now folded into Step 9d (NUM_COLS specialized on top of K-parallel base). The grouping microsteps (5b/5c) are now the full plan in Step 11 (vllm-style permute → per-expert grouped matmul → unpermute), with explicit correctness gates and the foundation routing-capture buffer broken out as Step 11a.

### Step 6: Revisit prompt-time attention and KV behavior

Primary files:

- `src/compute/forward.zig`
- `src/compute/attention.zig`
- `src/compute/elementwise.zig`
- `src/shaders/flash_attn.comp`
- `src/shaders/rope_fused.comp`

Look for decode-shaped prompt overhead:

- RoPE dispatched one token at a time
- per-token KV/cache writes with unnecessary synchronization
- flash attention invoked in a way that prevents prompt batching
- command buffer structure that forces too many submit/wait points during prompt ingestion

The goal here is not to invent a completely new runtime architecture in one jump. The goal is to eliminate obviously prompt-hostile execution patterns that should never have survived into the flagship RDNA path.

### Step 7: Keep the loop honest after every accepted step

After each accepted change:

- rerun the flagship prefill benchmark
- confirm prefill is still parseable and published
- confirm all required coherence models still pass
- refresh the phase budget by running once with `ZINC_PREFILL_PROFILE=1` and record the new biggest remaining bottleneck in the commit message

If the artifact loses `prefill_tps` again, treat that as a regression and stop. A fast path that the loop cannot observe is not a stable optimization program.

### Step 8: Wave64 K-parallel single-column DMMV (the precondition)

Primary files:

- `src/shaders/dmmv_q4k.comp`, `dmmv_q4k_batch.comp`
- `src/shaders/dmmv_q5k.comp`, `dmmv_q6k.comp`, `dmmv_q8_0.comp`
- `src/compute/dmmv.zig` (workgroup math)

Reference:

- `/Users/stepan/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl` (`K_PER_ITER`, subgroupAdd reduction)
- `mul_mat_vec.comp` (per-quant-type inner loop)

Why this is the precondition:

The current ZINC DMMV kernels assign **one wave64 thread per output row** and serially scan the K dimension inside that single thread. That leaves a 63× per-row parallelism gap. Multi-column batching, NUM_COLS specialization, and even Q8_1 mmq cannot recover this gap — they all multiply a starved dot product. llama.cpp's design instead has ~16 threads per row (or whatever divides cleanly into wave64 × NUM_ROWS), each thread processes `K_PER_ITER=8` quantized elements per iteration, and the row sum is computed with `subgroupAdd()` (or shared-memory fallback when `USE_SUBGROUP_ADD` isn't set).

Concrete microsteps:

- **Step 8a:** Rewrite `dmmv_q4k.comp` (single-column, no batch) to (rows-per-wg=2, threads-per-row=32) with `subgroupAdd()` reduction. Compare against the existing kernel on the LM head (single-dispatch sanity), then on the SSM proj loop (multi-dispatch reality). Foundation step — accept even if flat, because Step 9 cannot win without it.
- **Step 8b:** Repeat for `dmmv_q5k.comp`, `dmmv_q6k.comp`, `dmmv_q8_0.comp`. These are mechanical ports of 8a with different dequant snippets.
- **Step 8c:** Verify wave64 vs wave32 detection at pipeline-load time (RDNA4 is wave64 by default with `RADV_PERFTEST=coop_matrix`; older RDNA can be wave32). Pick a (rows, threads-per-row) tuple that divides evenly into both.

Done when:

- The flagship prefill benchmark moves by **≥ 8% on Q4_K-heavy projections** (signal: SSM proj phase from `ZINC_PREFILL_PROFILE=1` drops materially).
- Coherence sweep is green across all required RDNA models.
- Numerical sanity: row sums match the old kernel within reasonable f32 rounding tolerance.

### Step 9: NUM_COLS=2..8 specialized DMMV family (after Step 8)

Primary files:

- `src/shaders/dmmv_q4k_batch.comp`, `dmmv_q8_0_batch.comp`, plus a new `dmmv_q5k_batch.comp` if needed
- `src/compute/dmmv.zig` (`pipeline_q4k_batch_ncolN` array, dispatcher)
- `src/compute/forward.zig` (route SSM proj wqkv/z/alpha/beta and MoE router across multiple prompt tokens)

Reference:

- `/Users/stepan/Workspace/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:267` (`mul_mat_vec_max_cols = 8`)
- `mul_mat_vec_base.glsl` (NUM_COLS column-loop unroll)

Concrete microsteps:

- **Step 9a:** Re-introduce a `SPEC_NUM_COLS` GLSL specialization constant on top of Step 8's K-parallel kernel. (A prior LM-head-only attempt reverted because it didn't move the number; this time apply it to the multi-dispatch hot path.) Pre-compile NUM_COLS ∈ {1, 2, 4, 8} variants at startup. Skip 3/5/6/7 to keep the variant count manageable.
- **Step 9b:** Wire SSM proj (4 DMMVs per layer per token at `forward.zig:6552-6560`) to gather two adjacent prompt tokens' `norm_buf` columns into `attn_out_buf` (column-major), call `recordBatchDispatchPush(num_cols=2)`, then split outputs back to per-token paths. Flag-gate via `ZINC_PREFILL_BATCH=1` until coherence is green.
- **Step 9c:** Promote NUM_COLS=2 → NUM_COLS=4 → NUM_COLS=8 once 2 ships green. Each step doubles the cohort of prompt tokens that share one weight read.
- **Step 9d:** Apply the same pattern to MoE router DMMV (single Q4_K matvec of shape `M=n_experts, K=hidden`), which is identical in shape to SSM proj from the kernel's perspective.
- **Step 9e:** Apply to MoE gate/up/down. This requires more bookkeeping because each token's routed experts may differ — only tokens routed to the same expert can share that expert's weight read. This overlaps with Step 11 (grouped MoE); decide at Step 9d-completion time whether to proceed with Step 9e or pivot to Step 11.

Done when:

- Flagship prefill ≥ **+25%** vs Step 7 baseline.
- Coherence sweep green.

### Step 10: mul_mmq Q8_1 activation quantization path

Primary files (all new):

- `src/shaders/quantize_q8_1.comp` — fused per-block quantization of the prefill activation tensor into Q8_1 (32-element blocks: int8 values + f16 scale + f16 sum).
- `src/shaders/dmmv_q4k_q8_1.comp` (or `mul_mmq_q4k.comp`) — Q4_K weight × Q8_1 activation integer dot product. The integer dot product is the win: 2–4× faster than f32 dot on RDNA's int8 path.
- `src/shaders/dmmv_q5k_q8_1.comp`, `dmmv_q6k_q8_1.comp`, `dmmv_q8_0_q8_1.comp` — same pattern per weight quant.
- `src/compute/dmmv.zig` — new `pipeline_*_mmq` fields, new `recordMmqDispatch` helper, dispatch heuristic that picks mmq when `device.integer_dot_product` is supported and the dimensions divide cleanly.
- `src/compute/forward.zig` — quantize prefill input once per layer (or once per prefill chunk) into a scratch Q8_1 buffer; reuse it for every projection (attn QKV, SSM proj, MoE router, MoE gate/up/down).

Reference:

- `/Users/stepan/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/quantize_q8_1.comp`
- `mul_mmq.comp` + `mul_mmq_funcs.glsl`
- `ggml-vulkan.cpp:7856` (`should_use_mmvq` heuristic) and `:7975` (`ggml_vk_quantize_q8_1` call site)

Concrete microsteps:

- **Step 10a:** Add `quantize_q8_1.comp` plus a Zig wrapper that quantizes a `(K floats)` activation into a `(K/32 Q8_1 blocks)` device buffer. Validate by dequantizing back and comparing rounded values.
- **Step 10b:** Add `dmmv_q4k_q8_1.comp` (single-column, K-parallel, integer dot product using `iadd` accumulation then a final f32 rescale by `weight_scale × activation_scale`). Wire only the LM head as a sanity dispatch first.
- **Step 10c:** Wire SSM proj and MoE router through the mmq path. This is where the win shows up — both fire many times per prefill.
- **Step 10d:** Repeat for Q5_K, Q6_K, Q8_0 weights.

Done when:

- Flagship prefill ≥ **+40%** vs Step 9 baseline.
- Numerical sanity: per-token logits match the FP32 path within `1e-2` f32 RMS (acceptable Q8_1 quantization noise).
- Coherence sweep green across all required RDNA models, including the 7B/12B siblings (different N_HEADS, GQA group sizes — exercise the activation-quant path).

Risk note: If `device.integer_dot_product` is not enabled on the RDNA4 RADV stack we ship to, this step's payoff drops. Probe the device before sinking the full microstep budget — `ggml-vulkan.cpp` only enables mmq when the extension is present.

### Step 11: Grouped MoE GEMM (vllm-style permute → per-expert grouped matmul → unpermute)

Primary files:

- `src/compute/forward.zig` (the MoE block at `runDecodeStep` ~ `forward.zig:4608+` — replace the inner per-token-routed `pushDispatch4(..., Y=n_used, ...)` with a per-expert outer loop)
- New `src/shaders/moe_permute.comp` — scatter each (token, expert) pair into per-expert lanes
- New `src/shaders/moe_unpermute.comp` — gather per-expert outputs back into per-token order
- `src/shaders/dmmv_q4k_moe.comp` — modify (or add a `_grouped` variant) so each workgroup processes one expert's full token cohort instead of one token

Reference:

- `/Users/stepan/Workspace/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py` (`fused_experts_impl`, `invoke_fused_moe_kernel`)
- `/Users/stepan/Workspace/vllm/vllm/model_executor/layers/fused_moe/moe_permute_unpermute.py`
- `/Users/stepan/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp` (the `MUL_MAT_ID` row-gather path)

Why this is the largest remaining MoE win:

Today's GPU MoE path runs **one outer iteration per prompt token**, and inside each iteration all 8 routed experts run in parallel (Y workgroups in a single dispatch). The MoE weights (`gate_exps`, `up_exps`, `down_exps`) get re-read **once per prompt token** for every routed expert — and the L2 cache is only ~4 MB, far too small to hold a 35B model's expert weights across 154 tokens. A grouped GEMM amortizes one weight read across all tokens routed to that expert (~9.6 tokens on average for top-8 / 128 experts), which means **~10× lower DRAM bandwidth pressure** on the MoE phase. Per the current phase budget snapshot, MoE is ~1.6 s of the ~5 s prefill — so a 5× MoE speedup alone would buy roughly 0.8 s of latency back, which translates to ~+20–30% prefill tok/s.

Concrete microsteps:

- **Step 11a:** Build the per-(token, layer) routing capture buffer (referenced in earlier commits but not present in the working tree). Capture topk_ids[n_tokens, top_k] and topk_weights[n_tokens, top_k] into a device-side buffer that survives the per-token loop. Foundation step — accept even if flat.
- **Step 11b:** Implement `moe_permute.comp` (CPU permute is fine for the validation sanity pass; promote to GPU if CPU dispatch cost shows up in profile). Output: `permuted_inputs[total_routings, hidden_dim]`, `expert_first_token_offset[n_experts+1]`, `inverse_permutation[total_routings]`.
- **Step 11c:** Add `dmmv_q4k_moe_grouped.comp` — one workgroup per expert, processes all tokens routed to that expert. Reuses the K-parallel inner loop from Step 8.
- **Step 11d:** Wire MoE gate/up/down to use the grouped path. Skip experts with zero token count (cheap test).
- **Step 11e:** Implement `moe_unpermute.comp` — apply `inverse_permutation` and the topk weights to scatter expert outputs back into per-token positions, accumulate into `moe_out_buf`.

Done when:

- Flagship prefill ≥ **+35%** vs Step 10 baseline (or ≥ +25% standalone if Step 10 has not landed yet).
- Coherence sweep green; specifically, run the gpt-oss model that uses SOFTMAX_WEIGHT routing and Gemma 4 MoE that uses the per-expert scale tensor (ensure routing semantics survive the permute path).

Risk note: Grouped MoE has the highest correctness blast radius of any bet here. Misrouting a single token can pass the loop's coherence check but fail subtle outputs. Add a flag-gated CPU reference path that runs alongside on a single prompt and asserts per-token output L2 difference is below a tight threshold (`< 1e-4`) for the first few cycles after wiring.

### Step 12: RDNA shared-memory occupancy limiter

Primary files:

- `src/shaders/flash_attn.comp` (and any `dmmv_*.comp` that benefits)
- Pipeline creation in `src/compute/attention.zig` and `src/compute/dmmv.zig`

Reference:

- `/Users/stepan/Workspace/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:2990-2994` — RDNA-specific dummy LDS allocation (26–30 KB) on flash attention to force 4-subgroup-per-SIMD occupancy and avoid cache thrashing.

Concrete microstep:

- Add an unused `shared float dummy[ DUMMY_LDS_BYTES / 4 ];` declaration (or a specialization-constant-sized scratch array) to `flash_attn.comp` and to the K-parallel DMMV from Step 8, sized 26–30 KB on RDNA4. Validate that RADV reports 4 waves per SIMD via `RADV_DEBUG=preoptir,nir` or vendor profiler output (acceptable alternative: measure end-to-end and accept if positive).

Done when:

- Flash attention or DMMV phase from `ZINC_PREFILL_PROFILE=1` shows a measurable drop on RDNA4 (and no regression on the smaller RDNA models). Vendor-gate the LDS allocation if it harms non-AMD or non-RDNA4 paths.

### Step 13: Cooperative-matrix `mul_mm` for prefill projections

Primary files:

- `src/shaders/coop_matmul.comp` (extend tile size + dequant tiling)
- New `src/shaders/mul_mm_q4k.comp`, `mul_mm_q5k.comp`, etc. — quantized-weight tile loaders
- `src/compute/dmmv.zig` (`recordCoopMatmul` extension, dispatch heuristic that picks mul_mm when N > 8)
- `src/compute/forward.zig` (route projections that exceed DMMV's sweet spot — primarily once we batch ≥ 8 prompt tokens per dispatch)

Reference:

- `/Users/stepan/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp`
- `/Users/stepan/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_funcs.glsl` (per-quant-type tile loaders)

Concrete microsteps:

- **Step 13a:** Extend `coop_matmul.comp` from 16×16 tiles to BM=64, BN=32–64, BK=32 (matching llama.cpp's mul_mm). Validate on F16×F16→F32 first.
- **Step 13b:** Add `mul_mm_q4k.comp` — same coop-matrix outer loop, but the A-tile (weights) is loaded by a per-quant-type dequant function into shared memory rather than direct f16 load. Reference: `mul_mm_funcs.glsl:dequantFuncQ4K`.
- **Step 13c:** Wire the dispatch heuristic: when `num_cols ≥ 8` (i.e. prompt batch is large enough), pick mul_mm over the DMMV NUM_COLS=8 specialization. llama.cpp's threshold is `N > 8`.

Done when:

- Flagship prefill ≥ **+30%** vs Step 11 baseline (the gap narrows here because Steps 8/9/10 have already harvested a large fraction of the available win — coop matrix mainly helps when batched columns exceed ~8 tokens).
- Coherence sweep green.

Skip this step if Steps 8–11 already push us past Phase 3 (300 tok/s). It is the largest engineering investment per unit of throughput at that point in the curve.

## Success Criteria

This effort is succeeding when all of these are true:

- RDNA `qwen35-35b-a3b-q4k-xl` `context-long` has a real ZINC `prefill_tps` value in the artifact
- `loops/optimize_perf.ts` evaluates this effort primarily on `prefill tok/s`
- median total latency on the flagship long-context workload falls materially below the current `13002 ms`
- accepted changes keep all required RDNA coherence checks green
- the remaining gap to llama.cpp can be explained by a short, evidence-backed list of bottlenecks

## Non-Goals

- Do not treat tiny decode-only wins as success under this effort.
- Do not weaken coherence gates to keep a faster but wrong path.
- Do not spend cycles on Metal under this effort.
- Do not chase broad architectural rewrites before telemetry proves where the RDNA prompt path is losing time.
- Do not keep cosmetic barrier-scope narrowings that are within the noise band on RDNA4. See "Known flat" above.
- Do not add more phase profiling without a downstream structural change in the same cycle; the profile output already exists.
- Do not re-implement prefill embedding dequant layouts that were already tested and rejected (CPU / staging / interleaved variants — see Lessons section).
- Do not ship NUM_COLS=2..8 specialization on top of the current 1-thread-per-row DMMV shader. Step 8 (wave64 K-parallel inner loop) is a hard precondition.
- Do not wire `recordCoopMatmul` against quantized weights without either (a) an upfront f16 dequant scratch buffer that amortizes across many prompt tokens, or (b) a per-tile dequant-into-shared-memory loader (the `mul_mm_funcs.glsl` pattern from Step 13).
- Do not implement grouped MoE without a flag-gated CPU reference path running alongside for the first cycles (correctness blast radius is high).

## Likely Files

- `loops/efforts/MULTI_HOUR_EFFORT_6_RDNA_QWEN35_PREFILL.md`
- `loops/optimize_perf.ts`
- `tools/performance_suite.mjs`
- `tools/print_test_summary.ts`
- `src/main.zig`
- `src/compute/forward.zig`
- `src/compute/dmmv.zig`
- `src/compute/attention.zig`
- `src/compute/elementwise.zig`
- `src/compute/graph.zig`
- `src/vulkan/command.zig`
- `src/shaders/*.comp`
