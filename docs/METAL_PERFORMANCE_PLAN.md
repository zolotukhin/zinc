# Metal Performance Plan

Date: 2026-04-02
Model: `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`
Target GPU: local `Apple M4 Max` (`MTLGPUFamilyApple9`, 64 GiB RAM, 48 GiB recommended working set)

## Goal

Hit a stable local plain-decode path above `40 tok/s` on the M4 Max without
breaking correctness.

That is the right near-term target for the Apple Metal backend:

1. ZINC is already materially faster than the earlier Apple bring-up baseline
2. `40 tok/s` is only a modest step above the current local number
3. the current local gap to `llama.cpp` is real but not huge
4. this is now a kernel and memory-layout problem more than a generic host-plumbing problem

`40 tok/s` means `25.0 ms/tok`.

The current clean local baseline is about `36.52 tok/s`, or `27.4 ms/tok`.

So the practical problem is: remove about `2.4 ms` from the local token path.

## Progress Update: 2026-04-03

Phase 1 is now implemented and measured locally.

### Exact-shape Metal q8 benchmark added

New command:

```bash
zig build bench-metal-shapes -- \
  --model-id qwen35-35b-a3b-q4k-xl
```

This benchmarks the real local hot q8 shapes directly from the GGUF model:

- LM head `248320 x 2048`
- SSM qkv `8192 x 2048`
- SSM gate `4096 x 2048`
- SSM out `2048 x 4096`

### Exact-shape result: `dmmv_q8_0_k2048` is not a broad rollout win

Measured on April 3, 2026 on the same M4 Max with `200` timed iterations and
`25` warmup iterations:

- LM head: effectively flat, about `+0.17%`
- SSM qkv: worse, about `-9.45%`
- SSM gate: better, about `+5.47%`
- SSM out: not applicable (`K=4096`)

The benchmark also showed identical outputs between the generic and `k2048`
paths on the comparable cases.

Conclusion:

- do not roll out `dmmv_q8_0_k2048` broadly
- if we revisit it, do so only with selective shape gating

### Exact-shape dual SSM benchmark added

The single-shape q8 benchmark now has a real dual-SSM case:

```bash
zig build bench-metal-shapes -- \
  --model-id qwen35-35b-a3b-q4k-xl \
  --case ssm_dual --pipeline both
```

This measures the real paired SSM preprojection path:

- `8192 x 2048` `attn_qkv.weight`
- `4096 x 2048` `attn_gate.weight`

Measured on April 3, 2026:

- dual kernel (`dmmv_q8_0_dual`, `tg=512`): about `527.6 GB/s`
- separate single-q8 dispatches: about `511.8 GB/s`
- dual advantage: about `+3.0%`

Important follow-up sweep:

- dual `tg=256`: about `540.8 GB/s`
- dual `tg=512`: about `540.1 GB/s`
- dual `tg=1024`: about `572.8 GB/s`

So the dual kernel is real and profitable in isolation, but the best microbench
launch shape still did not translate into a whole-model decode win.

### Phase 2 started: attention-only Metal KV allocation

The Metal runtime now allocates K/V buffers only for the real full-attention
layers instead of all `40` layers.

Local end-to-end rerun on April 3, 2026:

```bash
zig build bench-metal -- \
  --model-id qwen35-35b-a3b-q4k-xl \
  --warmup 1 --runs 2 -n 128
```

Result:

- decode: `37.19–37.55 tok/s`
- average decode: `37.37 tok/s`
- decode time: `26.8 ms/tok`

This is a real improvement over the prior local baseline of about `36.52 tok/s`
and supports continuing down the memory-footprint path.

### Current worktree baseline is now about `38.11 tok/s`

Clean local rerun on April 3, 2026 in the current worktree:

```bash
zig build bench-metal -- \
  --model-id qwen35-35b-a3b-q4k-xl \
  --warmup 1 --runs 3 -n 128
```

Result:

- decode: `38.08–38.15 tok/s`
- average decode: `38.11 tok/s`
- decode time: about `26.2 ms/tok`

This is the current local number to beat.

### Exact-shape-guided global launch overrides did not beat baseline

Exact-shape sweeps suggested:

- LM head likes `q8_tg=512`
- dual SSM likes `q8_dual_tg=1024`

But the whole-model validation did not improve over the current baseline:

```bash
zig build bench-metal -- \
  --model-id qwen35-35b-a3b-q4k-xl \
  --warmup 1 --runs 3 -n 128 \
  --q8-tg 512 --q8-dual-tg 1024
```

Result:

- decode: `38.03–38.16 tok/s`
- average decode: `38.10 tok/s`

So the remaining Apple gap is not going to close with broad launch-size
overrides alone. The next win likely needs a more selective kernel change,
especially on LM head or the SSM out path, instead of another global threadgroup
sweep.

### Benchmark safety + profiling added

The local Metal benchmark tools now take the same per-GPU process lock as the
CLI/server path.

That matters because an accidental double-benchmark on the same M4 Max dropped
both runs to about `30 tok/s`, which would have completely poisoned the
comparison.

New benchmark capability:

```bash
zig build bench-metal -- \
  --model-id qwen35-35b-a3b-q4k-xl \
  --profile
```

This prints the Metal runtime profile from the benchmark path, which is useful
because the direct CLI profile path is not reliable in the current shell
environment.

### Explicit `f32` vs `q8_0` KV result: effectively flat

Clean local A/B on April 3, 2026 with `1` warmup run and `3` measured runs:

```bash
zig build bench-metal -- \
  --model-id qwen35-35b-a3b-q4k-xl \
  --warmup 1 --runs 3 -n 128 --kv-f32

zig build bench-metal -- \
  --model-id qwen35-35b-a3b-q4k-xl \
  --warmup 1 --runs 3 -n 128 --kv-q8
```

Result:

- `f32` KV: about `36.45 tok/s`
- `q8_0` KV: about `36.49 tok/s`

Conclusion:

- local Metal `q8_0` KV is currently neutral, not the decode win we wanted
- keep it as an optional path, not the default optimization bet

### Profile result: shared single-command decode is already active

Profiled benchmark on April 3, 2026:

```bash
zig build bench-metal -- \
  --model-id qwen35-35b-a3b-q4k-xl \
  --warmup 0 --runs 1 -n 32 --kv-f32 --profile
```

Key result:

- `shared_steps=37`
- `cmds=37`
- `commits=37`
- `gpu-moe=40.0 / step`
- `fallback-moe=0.0 / step`

That means the Metal backend is already using the single shared command buffer
path for this model. The next decode win is therefore less likely to come from
MoE control-flow cleanup and more likely to come from the hot `q8_0` kernels
themselves.

The same profile showed these local hot `q8_0` shapes over `37` decode steps:

- LM head `248320 x 2048`: `18.62 GiB`
- SSM preprojection `8192 x 2048`: `18.43 GiB`
- SSM gate `4096 x 2048`: `9.21 GiB`
- SSM out `2048 x 4096`: `9.21 GiB`

Important nuance:

- the `8192 x 2048` and `4096 x 2048` SSM tensors often travel through the
  dual-`q8_0` kernel together, so they are not fully independent optimization
  targets in the real decode path

### Dual-`q8_0` threadgroup sweep: short-run signal, long-run miss

With the new profile path, short `32`-token runs suggested `--q8-dual-tg 1024`
might be better than the default.

But the real `128`-token validation did not hold up:

```bash
zig build bench-metal -- \
  --model-id qwen35-35b-a3b-q4k-xl \
  --warmup 1 --runs 3 -n 128 --kv-f32 --q8-dual-tg 1024
```

Result:

- `36.34 tok/s` average decode

That is slightly worse than the clean default baseline, so this sweep is not a
keep.

## Current ZINC Local Baseline

Measured on April 2, 2026 on the local M4 Max with:

```bash
zig build bench-metal -Doptimize=ReleaseFast -- \
  --model-id qwen35-35b-a3b-q4k-xl \
  --warmup 1 --runs 2 -n 128
```

Clean benchmark result:

- decode: `36.46–36.58 tok/s`
- average decode: `36.52 tok/s`
- decode time: `27.4 ms/tok`
- prompt throughput: about `36.7 tok/s`

Profiled local run on the same date:

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zinc \
  --model-id qwen35-35b-a3b-q4k-xl \
  --prompt "The capital of France is" \
  -n 64 \
  --profile
```

Profile summary:

- `Generated 64 tokens in 1725.8 ms — 37.09 tok/s (27.0 ms/tok)`
- CPU embed: `0.30 ms` total
- CPU record: `18.60 ms` total, `0.270 ms/step`
- CPU sample: `13.28 ms` total, `0.208 ms/sample`
- submit/wait: `4020.74 ms` total, `58.272 ms/step`, `99.2%` of traced time

Dispatch-byte picture:

- `q8_0`: `130.93 GiB`, `74.5%`
- `q4_k`: `23.65 GiB`, `13.5%`
- `q5_k`: `15.19 GiB`, `8.7%`
- `q6_k`: `0.44 GiB`, `0.3%`

Path-byte picture:

- SSM: `69.00 GiB`
- attention: `18.61 GiB`
- routed MoE experts: `39.29 GiB`
- shared expert: `8.61 GiB`
- LM head: `34.72 GiB`
- router: `5.39 GiB`

Hottest local `q8_0` shapes:

1. LM head: `248320 x 2048`
2. SSM projection: `8192 x 2048`
3. SSM projection: `4096 x 2048`
4. SSM out: `2048 x 4096`

That is the local bottleneck until proven otherwise.

## Local llama.cpp Reference

Measured on April 2, 2026 on the same M4 Max and the same GGUF model using the
local Docker-provided `llama-server` binary:

```bash
/Users/zolotukhin/.docker/bin/inference/llama-server \
  --model-id qwen35-35b-a3b-q4k-xl \
  --host 127.0.0.1 \
  --port 8089 \
  --alias q \
  -c 4096 \
  -ngl all \
  -dev MTL0 \
  -ctk q8_0 \
  -ctv q8_0 \
  -b 4096 \
  -ub 1024 \
  -np 1 \
  -fa on \
  --reasoning-budget 0 \
  --no-webui \
  --perf \
  --temp 0
```

Warmup plus three raw-completions runs:

- run 1: `53.11 tok/s`
- run 2: `52.65 tok/s`
- run 3: `52.75 tok/s`
- average: about `52.83 tok/s`

Prompt throughput on the same runs:

- about `109.82–111.59 tok/s`

Local delta versus ZINC:

- decode gap: about `1.45x`
- prompt gap: about `3x`

Important runtime facts from the live llama.cpp log:

- `use fusion = true`
- `use concurrency = true`
- `use graph optimize = true`
- Metal residency sets enabled
- shared buffers enabled
- full model offloaded to `MTL0`
- KV cache stored as `q8_0`
- K/V total for this model config: `42.50 MiB`
- recurrent state buffer: `62.81 MiB`
- compute buffer reservation: `978.00 MiB`
- CPU output buffer: `0.95 MiB`

Important negative result:

- local `--no-repack` A/B was flat: `52.87 tok/s`

So repacking is not the main explanation for the local gap.

## What vLLM Does And Does Not Tell Us

As of April 2, 2026, the official vLLM docs say:

- Apple Silicon support is experimental on macOS and CPU-only
- Apple GPU inference is via the community `vllm-metal` plugin
- APC helps prefill, not decode

Relevant pages:

- <https://docs.vllm.ai/en/stable/getting_started/installation/cpu/>
- <https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/>

That means vLLM is not the primary source of local decode wins for this target.

For this Metal backend, llama.cpp is the more relevant comparison.

## What We Can Probably Copy From llama.cpp

### 1. Smaller and smarter Metal memory footprint

The local llama.cpp run is much more conservative with active decode memory:

- it stores K/V as `q8_0`
- it only allocates K/V for the real attention layers on this hybrid model
- it keeps a separate recurrent-state budget for the SSM layers

By contrast, ZINC currently allocates Metal K/V buffers for all `40` layers in
`f32`, even though this model only has `10` full-attention layers.

That is likely a real working-set penalty on Apple Silicon.

### 2. Better Metal `q8_0` decode kernels on the real hot shapes

The local ZINC profile says the backend is dominated by `q8_0` DMMV.

llama.cpp is faster on the same model without relying on prompt-cache tricks or
backend sampling.

So the most credible explanation is simply:

- its Metal `q8_0` matvec path is better on the exact decode shapes that matter

### 3. More aggressive Metal graph/runtime optimization

llama.cpp explicitly reports:

- fusion on
- concurrency on
- graph optimize on

ZINC already has a much better Apple path than before, but the prompt-side gap
suggests there is still graph-level win left in the Metal runtime.

## What We Should Not Chase First

These are not the first local decode blockers:

- vLLM APC or prefix-cache work
- backend sampling
- repack work
- broad threadgroup sweeps without exact-shape microbench evidence
- shared-expert dual-q8 reuse without a measured win

Already tried locally on April 2, 2026 and not kept:

- shared-expert dual `q8_0` gate/up reuse
- wider LM-head-only `q8_0` launch
- wider `ssm_out` `q8_0` launch
- generic `q8` threadgroup override sweeps

All of those were neutral or regressive against the local baseline.

## Highest-Value Local Opportunities

### A. Metal `q8_0` microbenching for the exact hot shapes

Add a dedicated Apple microbench path for:

- `248320 x 2048` LM head
- `8192 x 2048` SSM projection
- `4096 x 2048` SSM projection
- `2048 x 4096` SSM out

This should be the gate for any new `q8_0` kernel variant.

Success criterion:

- at least one hot shape improves clearly in isolation
- whole-model decode then improves by more than run-to-run noise

### B. Quantized Metal KV cache

Implement Apple-side `q8_0` K/V cache support.

This is directly supported by the local llama.cpp reference run and should:

- reduce working-set pressure
- reduce KV cache bandwidth
- make the Apple path closer to the reference setup

Success criterion:

- correctness preserved
- no prompt regression that outweighs decode gain
- measurable decode improvement on the 35B model

### C. Allocate K/V only for attention layers

This model has full attention every fourth layer, not every layer.

The Metal backend should not allocate K/V buffers for SSM-only layers.

This is a simpler memory-footprint win than a full new kernel path and should
be done even if `q8_0` KV takes longer.

Success criterion:

- lower Metal memory footprint
- no regression
- ideally improved decode stability under pressure

### D. GPU-side greedy argmax for Metal

ZINC still copies the full logits vector back to CPU and scans it in
`sampleGreedy(...)`.

That is not the main current gap, but it is still avoidable tail work.

It is a second-order item after the `q8_0` and KV work, not before.

Success criterion:

- remove full-vocab logits copy on the fast path
- preserve exact greedy output
- measurable, even if small, token-tail reduction

### E. Metal residency / working-set planning

ZINC already exposes:

- `recommendedMaxWorkingSetSize`
- unified-memory detection
- private-buffer support

But it does not yet have a real residency policy comparable to the local
llama.cpp runtime behavior.

The first step is not full Metal-4 feature work. The first step is:

- use working-set-aware planning for model, K/V, and scratch buffers
- avoid holding more decode-resident state than the model actually needs

## Execution Order

### Phase 1. Add exact-shape Metal hot benchmarks

Objective:

- stop guessing on Apple `q8_0`
- measure only the real local shapes

Deliverables:

- a Metal hot-bench target or mode
- stable shape-level numbers for the four hot local `q8_0` cases

### Phase 2. Shrink the Metal decode memory footprint

Objective:

- copy the clear llama.cpp memory-side wins first

Deliverables:

- attention-layer-only K/V allocation
- measured before/after memory footprint
- measured before/after decode throughput

### Phase 3. Implement `q8_0` K/V on Metal

Objective:

- make the Apple fast path closer to the local reference configuration

Deliverables:

- `q8_0` K/V storage and access path
- correctness validation
- benchmark data on the 35B model

### Phase 4. Rework the hottest Metal `q8_0` kernels

Objective:

- improve the exact kernels the local profile says dominate

Deliverables:

- shape-driven `q8_0` kernel changes
- microbench win first
- whole-model win second

### Phase 5. Remove remaining token-tail waste

Objective:

- reduce non-kernel per-token overhead after the main bandwidth work lands

Deliverables:

- GPU greedy argmax
- less full-logits readback
- updated profile numbers

## Keep / Revert Rules

For Apple Metal changes, keep the bar simple:

1. do not keep regressions
2. prefer a 3-run average over a single lucky run
3. require correctness plus a real benchmark win
4. prefer exact-shape microbench wins before whole-model rollout for `q8_0`

## Current Recommendation

The next concrete Apple work should be:

1. add an exact-shape Metal `q8_0` hot-bench
2. stop allocating `f32` K/V for all `40` layers
3. implement `q8_0` K/V on Metal
4. only then revisit GPU argmax or other tail cleanup

Updated next step after April 3 measurements:

1. add an exact-shape benchmark for the dual-`q8_0` SSM preprojection path
2. benchmark new kernels against the real hot shapes, not generic launch sweeps
3. focus on the three dominant decode-side `q8_0` buckets:
   `lm_head`, dual SSM preprojection, and `ssm_out`

That is the shortest path with a real chance of moving the local backend from
`36.5 tok/s` into the `40 tok/s` range.
