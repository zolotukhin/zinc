# Optimization 4: Prefill Recovery vs llama.cpp

## Current State (2026-04-09)

Target model: `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` on the RDNA4 node.

The current public benchmark artifact in `site/src/data/zinc-performance.json` shows a severe long-context prefill gap on the `context-long` scenario:

- ZINC prefill: `73.7 tok/s`
- llama.cpp prefill: `895.2 tok/s`
- Relative: `8.23%` of baseline
- Prompt length: `154` prompt tokens
- ZINC total latency: `12684.9 ms`
- llama.cpp total latency: `384.8 ms`

Decode is not the main problem in this workload:

- ZINC decode: `81.1 tok/s`
- llama.cpp decode: `111.5 tok/s`

The bottleneck is prompt processing, not next-token generation.

## Goal

Recover a large fraction of the missing prefill throughput on RDNA4 without regressing coherence.

Practical milestone targets for the same `context-long` benchmark:

- Phase 1 target: `>= 150 tok/s`
- Phase 2 target: `>= 250 tok/s`
- Phase 3 target: `>= 400 tok/s`
- Stretch target: `>= 600 tok/s`

Even `400 tok/s` would cut a major part of the user-visible wait and prove that prefill is no longer structurally broken.

## Benchmark Contract

This effort is scored on `prefill tok/s`, not decode tok/s.

Use the same long-context prompt shape that feeds the site benchmark:

- long reference packet
- answer near the end
- `154` prompt tokens in the current artifact
- `-n 8` decode cap only to complete the run

Critical benchmark conditions:

- `zig build -Doptimize=ReleaseFast`
- `RADV_PERFTEST=coop_matrix`
- benchmark the RDNA node with no competing `zinc` or `llama-*` GPU users
- keep the coherence gate active across the supported model set after every accepted change

## Why The Gap Is So Large

This is too large to be explained by one small shader inefficiency. Assume a stack of issues until measurement disproves it:

1. The current prefill path is still largely token-serial, so weights are reread for each prompt token.
2. MoE and attention work may be falling back to decode-style dispatch patterns during prompt ingestion.
3. We likely lack phase-level timing inside prefill, so regressions hide inside one aggregate `Prefill:` line.
4. The historical optimization loop was scoring decode and using non-ReleaseFast settings, so it could not reliably drive prefill work.

## Execution Order

### Step 1: Lock the benchmark harness

Files:

- `loops/optimize_perf.ts`
- `loops/optimize_zinc.ts`
- `loops/optimize_perf.test.ts`
- `loops/optimize_zinc.test.ts`

Done when:

- the effort is scored on `prefill tok/s`
- the loop uses `ReleaseFast`
- the loop runs ZINC with `RADV_PERFTEST=coop_matrix`
- the benchmark prompt matches the site’s long-context workload closely enough to compare numbers directly

This step is infrastructure, but it prevents another week of optimizing the wrong metric.

### Step 2: Add prefill stage timing, not just one summary line

Primary file:

- `src/compute/forward.zig`

Add timing around the major prefill buckets:

- embedding upload
- layer norms / elementwise
- attention projections
- attention kernel
- MoE router
- expert gate/up
- expert down
- shared expert
- KV/cache writes
- final norm + LM head

Requirements:

- log one compact per-run phase summary behind a debug or profile guard
- no per-token spam in the steady benchmark path
- the output should make it obvious whether prefill is dominated by DMMV, MoE routing, or token-serial attention plumbing

Without this, we are still guessing.

### Step 3: Verify whether prefill is structurally token-serial

Primary files:

- `src/compute/forward.zig`
- `src/compute/dmmv.zig`
- `src/compute/elementwise.zig`
- `src/compute/attention.zig`

Answer these questions with code and measurement:

- Are prompt tokens still driven through the same single-token decode path?
- Which tensor families already have batch-capable dispatch support?
- Which hot layers still force `N` separate weight reads for `N` prompt tokens?

Exit condition:

- a short note added to this effort doc or commit message describing which subsystems are already batch-capable and which are still serial

### Step 4: Harvest the low-risk batch wins first

Do not start with the hardest MoE grouping problem if simpler wins are still missing.

Prioritize:

- batched RMS norm / elementwise where the kernel already supports `n_tokens`
- batched QKV / projection DMMV where the weight matrix can be read once for multiple prompt columns
- batched embedding upload / staging reuse
- reducing command-buffer churn across prompt tokens

Acceptance:

- measurable prefill gain on the benchmark prompt
- no coherence regressions on Qwen, Gemma, or GPT-OSS

### Step 5: Attack the real MoE prefill cost

Qwen 3.5 35B A3B is MoE-heavy. If we batch only non-MoE work, we will still leave a large ceiling in place.

Expected work:

- batch router projection
- collect per-token top-k expert choices
- group tokens by expert
- run gate/up/down for gathered token blocks
- scatter expert outputs back into token order

Start with a simple grouping design that is easy to validate before chasing a perfect schedule.

Validation rule:

- correctness first
- then throughput

Wrong expert grouping will look fast and silently destroy output quality.

### Step 6: Revisit attention/KV behavior in prefill

Even with DMMV improvements, prefill will stay weak if prompt-time attention still pays decode-style overheads.

Check for:

- excessive per-token command submissions
- avoidable barriers between prompt tokens
- redundant KV/cache write plumbing
- RoPE dispatch patterns that block batching

The goal is not to invent a full new graph immediately. The goal is to remove obviously decode-shaped prompt execution.

### Step 7: Re-benchmark against llama.cpp after each accepted milestone

Do not judge success only against the previous ZINC run. Track the remaining headroom against llama.cpp:

- prefill tok/s
- total latency
- end-to-end throughput

The real question is not “did ZINC gain 5 tok/s?”.
The real question is “did the gap to baseline meaningfully shrink?”.

## Success Criteria

The effort is a success when all of these hold:

- the loop benchmarks prefill correctly
- the site benchmark and the loop report comparable prefill numbers
- Qwen3.5-35B `context-long` prefill is materially higher than `73.7 tok/s`
- no accepted commit regresses coherence on the supported model set
- the remaining gap is attributable to a small number of named bottlenecks rather than “prefill is slow”

## Non-Goals

- Do not chase tiny decode-only wins under this effort.
- Do not widen architecture support here.
- Do not weaken coherence gates to keep a faster but wrong prefill path.

## Likely Files

- `src/compute/forward.zig`
- `src/compute/dmmv.zig`
- `src/compute/elementwise.zig`
- `src/compute/attention.zig`
- `src/vulkan/command.zig`
- `src/vulkan/pipeline.zig`
- `src/shaders/*.comp`
- `loops/optimize_perf.ts`
- `loops/optimize_zinc.ts`
