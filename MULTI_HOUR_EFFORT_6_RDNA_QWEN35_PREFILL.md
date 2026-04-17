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

- per-phase prefill timing on the RDNA path
- hot-kernel or per-layer timing for the prompt path
- llama.cpp reference re-check when a milestone lands

## What The Current Evidence Already Says

Several things are already clear from the current benchmark artifact:

1. The prompt-time bottleneck is not explained by a modest decode deficit. ZINC is slower than llama.cpp on decode, but not by the factor implied by `13002 ms` vs `232 ms` total latency.
2. The long-context prompt is not huge. This is not a million-token scaling issue. It is a medium prompt that should already be cheap on a healthy prefill path.
3. We should assume some part of RDNA prefill is still structurally decode-shaped until measurement proves otherwise.
4. The missing `prefill_tps` field for the flagship scenario is itself a regression hazard. If the loop cannot observe the metric reliably, it will drift back toward decode-oriented optimization.

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

Add timing that answers where prompt time is actually going:

- embedding upload and staging
- layer norm and small elementwise passes
- QKV or projection DMMV time
- flash attention time
- KV/cache write time
- router projection time
- expert gate/up time
- expert down time
- shared expert time
- final norm and LM head
- command submission / synchronization overhead if it is visible

Constraints:

- one compact summary per benchmark run
- no huge per-token log spam in normal runs
- enough detail to compare one cycle to the next

This step is complete when one stalled benchmark tells us which bucket owns most of prompt time.

### Step 3: Prove which parts of prefill are still token-serial

Primary files:

- `src/compute/forward.zig`
- `src/compute/graph.zig`
- `src/compute/dmmv.zig`
- `src/compute/elementwise.zig`
- `src/compute/attention.zig`

Questions to answer:

- is prompt ingestion still traversing the decode loop token-by-token in the hot sections?
- which ops already support `n_tokens > 1` and which ones collapse back to one token per dispatch?
- where are we rereading large weight tensors once per prompt token?
- where are command buffers or barriers being rebuilt per token when they could be amortized?

Record the answer in the effort log or commit message after the measurement cycle. Future agents should not have to rediscover this.

### Step 4: Take the safest batch wins first

Start with improvements that reduce prompt overhead without changing model semantics aggressively.

Likely targets:

- batched embedding upload or staging reuse
- batched RMS norm and elementwise passes that already understand `n_tokens`
- projection paths where one weight read can serve multiple prompt columns
- fewer per-token command submissions
- fewer redundant barriers across prompt tokens

Acceptance rule:

- measurable prefill gain on the flagship benchmark
- coherent output still passes across all RDNA models

If these steps do not move the number, that is evidence. Use it to justify deeper MoE work, not as a reason to keep random churn.

### Step 5: Attack Qwen35 MoE prefill directly

Primary files:

- `src/compute/forward.zig`
- `src/compute/dmmv.zig`
- `src/shaders/*.comp`

Qwen3.5-35B A3B is MoE-heavy. Prefill will stay capped if expert work is still scheduled like decode.

Likely work:

- batch router projection for prompt tokens
- collect per-token top-k routing for a prompt chunk
- group tokens by expert
- run gate/up/down over gathered token blocks
- scatter the accumulated expert outputs back into original token order
- keep correctness checks tight, because expert misrouting can look fast and silently poison outputs

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
- record the new biggest remaining bottleneck

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
