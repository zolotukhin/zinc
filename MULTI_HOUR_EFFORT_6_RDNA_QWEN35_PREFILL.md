# Optimization 6: RDNA Prefill Recovery for Qwen3.5-35B

## Current State (2026-04-16)

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

## Evidence So Far (cycles 1–24)

This section is not history for its own sake. It is the set of facts the next cycle must start from so we stop relitigating settled questions.

### What has actually moved the number

Only three cycles out of 24 produced a real best-checkpoint improvement. All three share the same shape: restructure *when* work happens, not *how* a single dispatch is tuned.

- Cycle 4 (+~3%): skip `final_norm` + `LM_head` and the last-layer FFN/MoE body for non-terminal prefill tokens. Dead work elimination.
- Cycle 5 (+~7%): double-buffered prefill command buffer + embedding staging. The CPU records token N+1 while the GPU runs token N. Overlap, not per-dispatch tuning.
- Cycle 20 (+~2%): extend cycle 4's skip into the full last-layer attention block (Q/gate DMMV, Q-norm, Q-RoPE, flash_attn, sigmoid, O-proj, residual) for non-terminal prefill tokens on hybrid Qwen3.5-35B. More dead work elimination.

Best checkpoint as of cycle 24: `25.67 tok/s` prefill. Target: `150 tok/s` (Phase 2), `300 tok/s` (Phase 3). We are roughly one sixth of the way to Phase 2.

### What has been proven flat on RDNA4 — do not try again without new evidence

The following category has been tried seven times (cycles 9, 17, 18, 19, 21, 22, 24 of the first run) with cumulative movement below 0.25 tok/s. On RDNA4 + RADV, narrowing a single back-to-back `computeBarrier` to a buffer-scoped barrier does not unlock a measurable win. Two of these cycles foundation-kept flat, the rest reverted flat.

- Single-buffer or multi-buffer `computeBarrier` narrowings between successive compute dispatches.

Stop proposing cosmetic variations of this pattern. If a future cycle attacks barriers, it must either (a) remove a barrier outright by restructuring what reads what, or (b) come with a micro-benchmark showing that a specific RADV path responds differently than cycles 9/17/18/19/21/22/24 did.

### Second-run finding: pair-dispatch (num_cols=2) via the existing batch shaders is net-negative on RDNA4

Cycles 1, 3, 5, 7, 9 of the second run landed flag-gated pair-batch infrastructure across SSM proj + MoE router + layer-0 peek-ahead. Cycle 8 and cycle 9 measured the flag-ON path and found:

- Cycle 8 (duplicate col 0 == col 1): −0.12 tok/s, SSM proj +23 ms.
- Cycle 9 (rewrote `dmmv_q8_0_batch.comp` with proper wave64 parallelism, 2 rows/WG × 64 threads + subgroupAdd): still −0.8 tok/s with flag on.

Root cause, as the agent diagnosed: even with the shader parallelism fixed, the per-layer "stage norm into col0/col1, split 4 outputs back, barrier between" chain costs more than the weight-read amortization saves. Back-to-back single-column DMMVs win via the L2 weight cache.

**Do not propose another num_cols=2 variant on top of the existing `dmmv_q8_0_batch` / `dmmv_q4k_batch` shaders.** The fix is architectural, not incremental:

- Port llama.cpp's compile-time specialized DMMV design: pre-compile 8 variants of the DMMV shader with `NUM_COLS` baked in as a GLSL specialization constant (values 1..8), so each variant has its inner loop unrolled and the register allocator sees a static column count. See `/Users/zolotukhin/Workplace/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` line 267 (`mul_mat_vec_max_cols = 8`) and `vulkan-shaders/mul_mat_vec_base.glsl`.
- Route to a proper matmul kernel (mul_mm-style, preferably cooperative-matrix via the existing `src/shaders/coop_matmul.comp`) when N exceeds the DMMV sweet spot. Llama.cpp uses N > 8.
- Quantize activations to Q8_1 once per prefill chunk, then use the `mul_mmq` pattern for the largest prefill DMMVs. Reference: `/Users/zolotukhin/Workplace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/quantize_q8_1.comp` and `mul_mmq.comp`.

### Second-run finding: 3-deep prefill pipeline is flat

Cycle 2 extended the double-buffered prefill pipeline from 2-deep to 3-deep and measured no change. Submit/wait is already saturated at 2-deep on the 154-token workload; the record+submit+wait gap is not the dominant cost. Do not re-attempt 4-deep without first proving that the CPU record time has materially grown (e.g. from expert-grouping bookkeeping in Step 5b/5c).

### Dormant infrastructure currently in the tree from the second run

These commits are in HEAD and they are net-negative or dormant per the findings above. A pivot cycle should decide whether to revert them or reuse parts of them:

- Cycle 1 (`20c0ea8f`): `recordBatchDispatchPush` helper. Probably reusable — a specialized 8-variant DMMV would still want a push-descriptor dispatch helper.
- Cycle 3 (`74f9ff47`): SSM wqkv pair-dispatch wiring. Dead-end as shaped; revert unless a follow-up cycle repurposes it for a specialized variant.
- Cycle 5 (`b035b168`): extended the pair-dispatch wiring to all 4 SSM projections. Same status as cycle 3.
- Cycle 7 (`b22ce68e`): layer-0 peek-ahead with real cross-token data. Clever but unsuccessful — the barrier-and-copy chain still costs more than the amortization saves.
- Cycle 9 (`d3a968d`): wave64-parallel rewrite of `dmmv_q8_0_batch.comp`. Useful as a reference for what the parallelism should look like, but the multi-column variant is still slow. A per-num_cols-specialized shader would subsume it.

### What has been repeatedly attempted and rejected as pointless in isolation

- "Add more phase profiling" without a downstream change in the same cycle. Cycles 12 and 16 added phase/dispatch counters and were reverted because they did not touch tok/s. The profile line *already* exists behind `ZINC_PREFILL_PROFILE=1`; a cycle should turn that flag on, read the output, then act.
- "Restructure where prompt embedding dequant happens." Cycles 14, 15, 23 explored CPU / staging / interleaved variants of the same idea. The current checked-in design (upfront bulk dequant into host-mapped staging, cycles 14+15) is the accepted equilibrium. Do not resubmit interleaving or async-thread variants without a new bottleneck proof.

### Current phase budget snapshot (from cycle 3 measurement, refresh before acting)

Numbers below are for the flagship long-context benchmark. They are a snapshot; the loop should re-run with `ZINC_PREFILL_PROFILE=1` before starting a new structural attack so the agent targets the real largest bucket instead of a stale one.

- attention: ~0.7 s
- MoE (router + topk + gate_up + swiglu + down + weighted_acc): ~1.6 s
- shared expert: small, typically <0.1 s
- SSM (proj + conv + delta + gnorm + out): ~1.8 s — **largest single bucket**
  - ssm_proj alone: ~1.3 s — the single largest sub-bucket
- final tail (norm + LM head): ~0.1 s (cycle 4 already skips this for non-terminal tokens)
- CPU + submit/wait gap after GPU phases sum: ~0.5–0.8 s

The fact that SSM proj is the single largest bucket and has never been batched across prompt tokens is the loudest unexploited signal in the effort.

## Dormant Infrastructure Already In The Tree

These code paths were added in earlier cycles and compile into the binary, but have **zero callers in the prefill hot path**. Wiring them in is lower risk than inventing new infrastructure, and every cycle that tries a fresh micro-optimization without wiring them first is leaving free throughput on the floor.

- `DmmvDispatch.recordBatchDispatch` in `src/compute/dmmv.zig` — multi-column DMMV entry point supporting `num_cols > 1`. Zero callers anywhere in the codebase.
- `DmmvDispatch.pipeline_q4k_batch` / `pipeline_q8_0_batch` — loaded pipelines that are only invoked today with `num_cols = 1` (and only for the large LM head in `forward.zig`). The multi-column shader path exists but has never been exercised against prompt tokens.
- `CommandBuffer.computeBuffersBarrier` (cycle 21) — multi-buffer barrier helper. Useful when a cycle structurally removes a global barrier (not when it cosmetically narrows one).

Any cycle whose self-analysis proposes "batch X across prompt tokens" and does not reference the existing `recordBatchDispatch` helper is proposing unnecessary new infrastructure. Check for the existing helper first.

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

### Step 2.5: Wire up the dormant batch DMMV infrastructure before inventing new paths

This is the highest-leverage remaining work that is structurally easier than it looks, because the shader, pipeline, and Zig entry point already exist:

- `recordBatchDispatch` in `src/compute/dmmv.zig` accepts `num_cols` and supports Q4_K and Q8_0 via dedicated batch shaders (and falls back to per-column DMMV for other quant types). Zero production callers.
- The simplest wiring: pick one prefill DMMV on the hot path (start with SSM proj because it is the biggest single bucket), accumulate two adjacent prompt tokens' inputs into an adjacent-column staging layout, call `recordBatchDispatch` with `num_cols = 2`, and split the output into the two per-token paths that downstream dispatches expect.
- Correctness gating: keep the single-column path as a fallback, flag-gate the new path with a runtime env like `ZINC_PREFILL_BATCH=1` until the coherence sweep is green, then remove the flag once it sticks.
- Do not open this step by writing a new shader. The existing `dmmv_q8_0_batch.comp` and `dmmv_q4k_batch.comp` already handle `num_cols` correctly.

Success criterion: a cycle lands that wires `recordBatchDispatch` into at least one prefill DMMV call site with `num_cols ≥ 2` and a flagship coherence sweep stays green. Even if the first wiring is flat on tok/s, it is a foundation step that unblocks Steps 4/5.

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
- projection paths where one weight read can serve multiple prompt columns (hook into Step 2.5)
- fewer per-token command submissions, ideally with the double-buffered pipeline-prefill path extended to 3+ deep
- fewer redundant barriers across prompt tokens — only accept if the barrier is actually *removed*, not just narrowed in scope

Acceptance rule:

- measurable prefill gain on the flagship benchmark (>= the usual threshold versus the best checkpoint)
- coherent output still passes across all RDNA models

If these steps do not move the number, that is evidence. Use it to justify deeper MoE work, not as a reason to keep random churn.

### Step 5: Attack Qwen35 MoE prefill directly

Primary files:

- `src/compute/forward.zig`
- `src/compute/dmmv.zig`
- `src/shaders/*.comp`

Qwen3.5-35B A3B is MoE-heavy. Prefill will stay capped if expert work is still scheduled like decode. Break this into three concrete microsteps so the loop can pick one per cycle without having to invent the whole plan.

- **Step 5a: Batch router projection across prompt tokens.** The router is a single Q4_K DMMV per token of shape `M=num_experts, K=hidden`. Two adjacent tokens can share the same weight read by issuing one `recordBatchDispatch` with `num_cols = 2`. This is the smallest MoE batching step and keeps all downstream routing/topk logic per-token. Enable via the same `ZINC_PREFILL_BATCH` flag from Step 2.5.
- **Step 5b: Collect per-token top-k routing for a prompt chunk and group tokens by expert.** For a chunk of N prompt tokens, compute the N×top_k routing vector, invert it into per-expert token lists, and pad. This is pure CPU/GPU bookkeeping; no new kernels.
- **Step 5c: Run per-expert gate/up/down over the gathered token block.** For each active expert, run one larger matmul over its gathered token inputs instead of N tiny per-token matvecs. Scatter the accumulated expert outputs back into original token order using the inverse of the permutation from 5b. The gather/scatter can be on the GPU (elementwise kernel) or CPU-side initially for validation.

Guidance:

- Start with Step 5a. It is the smallest MoE change that exercises the shared-weight code path, and it composes with Step 5b/5c when they land.
- Step 5b can ship as a foundation step even if Step 5c is not yet wired; the routing arrays are inputs that no one reads yet.
- Keep correctness checks tight. Expert misrouting can look fast and silently poison outputs. Run the full coherence sweep, not just the one-prompt sanity check.

Start with the simplest grouping design that is easy to validate. A mediocre but correct grouped path is worth more than an ambitious schedule that breaks coherence.

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
- Do not re-implement prefill embedding dequant layouts that were already tested and rejected (cycles 14/15/23).

## Likely Files

- `MULTI_HOUR_EFFORT_6_RDNA_QWEN35_PREFILL.md`
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
