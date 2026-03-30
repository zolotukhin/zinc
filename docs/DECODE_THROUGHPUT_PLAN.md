# Decode Throughput Plan

Date: 2026-03-30
Model: `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`
Target GPU: RDNA4 test node (`AMD Radeon Graphics (RADV GFX1201)`, `576 GB/s`, `64 CUs`)

## Goal

Keep the clean ReleaseFast decode path stably above `30 tok/s`, push the chat/reasoning path above `30 tok/s`, and improve aggregate GPU utilization without breaking correctness.

This plan separates two different goals that were previously mixed together:

1. `Single-stream raw decode latency / tok/s`
2. `Reasoning chat latency / tok/s`
3. `Aggregate GPU bandwidth utilization`

Those are related, but they are not the same target. A single decode stream is not expected to saturate `576 GB/s` of DRAM bandwidth on this workload.

## Measured Baseline

Measured on March 30, 2026 on the RDNA4 test node from the current checkout synced to `/root/zinc`.

### Current clean-node result

After clearing the RDNA4 node of competing `zinc` and `llama` processes and re-running the current checkout:

- clean `ReleaseFast` CLI run: `33.58 tok/s`
- raw `/v1/completions` on the same server path: `33.55 tok/s`
- raw `/v1/completions` at `concurrency=4`: `33.98 tok/s` aggregate
- one longer reasoning chat sample: `28.40 tok/s`
- modeled decode bandwidth at `33.58 tok/s`: `112.5 GB/s`, about `19.5%` of `576 GB/s`

This means the current tree already exceeds the old `15 tok/s` target and also clears the new `30 tok/s` target on the raw decode path.
The remaining gap is no longer "how do we get above 15"; it is "how do we keep reasoning chat above 30 and drive utilization higher without regressing quality."

### Clean CLI run

Command shape:

```bash
zig build -Doptimize=ReleaseFast
RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
  -m /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt "The capital of France is"
```

Observed on the cleaned node:

- `Generated 256 tokens in 7624.7 ms`
- `33.58 tok/s`
- `29.8 ms/tok`

### Current interpretation

- The raw decode problem is no longer "stuck at 7 tok/s"
- The next performance problem is specifically the reasoning chat path and aggregate GPU utilization
- `--profile` still needs work in `ReleaseFast`; it is too intrusive for leaderboard-style comparisons

## What The Current Numbers Mean

The runtime currently prints a "Bandwidth" line after decode, but that metric only models:

- final RMS norm
- LM head projection
- logits readback

It does not represent full-token decode traffic. Treating it as whole-model memory utilization is incorrect.

The decode graph artifact at `/tmp/zinc-decode-graph.json` models roughly `3.35 GB/token` for this graph. Using that model:

- `28.40 tok/s` implies about `95 GB/s`
- `33.58 tok/s` implies about `112.5 GB/s`
- `112.5 GB/s` is only about `19.5%` of `576 GB/s`

Conclusion:

- `30 tok/s` is plausible for a single stream and already achieved on the raw decode path
- the remaining target is the chat/reasoning path, not basic decode throughput
- `100%` DRAM-bandwidth utilization is not a realistic single-stream target here
- if the real goal is to drive memory bandwidth higher, that requires batching or concurrent decode streams

## Current Measurement Problems

### 1. `--profile` is coupled to deep diagnostics

`--profile` currently enables:

- Vulkan timestamps
- extra CPU/GPU reference checks
- mid-token readbacks
- diagnostic command-buffer flushes
- SSM debug readback
- per-layer hidden-state summaries

That makes the profile result intrusive and incomplete at the same time.

### 2. The profile summary is not full-token

The current timestamp summary is printed after the batched layer work submission, before:

- final RMS norm
- LM head
- logits readback
- sampling

So the printed GPU time is not the full decode token.

### 3. The optimization loop is using the wrong KPI

The loop currently measures with `--debug`, so it is optimizing a path that is slower than clean inference by about `21%`.

### 4. The bandwidth line is misleading

The current runtime prints a small "effective GB/s" number that looks like whole-token bandwidth utilization, but it is only an LM-head-centric estimate.

## Execution Plan

### Phase 1. Fix measurement before touching kernels

Objective:

- make `--profile` low-perturbation
- report full-token timing boundaries
- benchmark clean decode in the optimization loop
- stop presenting the LM-head-only metric as whole-decode bandwidth

Concrete work:

1. Split lightweight profiling from deep diagnostics
2. Move profile summary to the real end of the token
3. Gate expensive validation readbacks behind an explicit diagnostics flag, not normal profiling
4. Rename or replace the current bandwidth line so it is not mistaken for whole-token utilization
5. Change the optimization loop to benchmark clean mode by default

Expected result:

- clean benchmark stays around `10.5-11 tok/s`
- profiling overhead drops materially from the current `5.52 tok/s` path
- future optimization work gets trustworthy before/after measurements

### Phase 2. Remove obvious per-token tail overhead

Objective:

- reduce CPU-visible tail work after GPU decode

Concrete work:

1. Move greedy argmax fully onto the GPU using the existing `argmax` shader path
2. Stop copying full logits back for greedy sampling
3. Keep full logits readback only for explicit diagnostics modes

Expected result:

- small but real single-stream gain
- lower per-token CPU synchronization pressure

### Phase 3. Reduce Vulkan setup overhead

Objective:

- reduce repeated descriptor and command-buffer work per token

Concrete work:

1. preallocate and reuse descriptor sets where bindings are stable
2. avoid rebuilding descriptor state that does not change per token
3. move closer to replayable decode instead of hot-path rebuilds

Expected result:

- likely the largest near-term latency win after measurement cleanup

### Phase 4. Fuse same-input decode work

Objective:

- reduce repeated reads of the same normalized vector

Concrete work:

1. fuse MoE `gate` and `up` projections when they consume the same FFN-normalized input
2. evaluate attention/SSM projection fusion where one normalized vector fans out to multiple DMMVs

Expected result:

- better effective memory locality
- fewer dispatches and barriers
- best path from `10-11` toward `12-15 tok/s`

### Phase 5. Retune kernels by actual decode shapes

Objective:

- optimize where decode spends time, not just where the largest tensor exists

Hot decode shapes to measure:

- `248320 x 2048` LM head
- `8192 x 2048`
- `4096 x 2048`
- `512 x 2048`

Expected result:

- shape-specific DMMV improvements
- better understanding of where occupancy is falling off

## Success Criteria

### Short term

- clean no-debug CLI benchmark reproducibly above `15 tok/s`
- profile mode stays near clean-mode throughput
- loop benchmarks clean decode instead of debug decode

### Medium term

- stable `20+ tok/s` on the RDNA4 node for this model
- profile output identifies full-token GPU time with low perturbation

### Long term

- `25+ tok/s` single stream
- separate server-mode work for aggregate throughput and higher bandwidth utilization via concurrency

## Notes

- The loop appearing "stuck at 7 tok/s" was partly a tooling problem, not only a kernel problem.
- The current clean build now exceeds `22 tok/s` on the idle RDNA4 node; the next step is to make that the default measured baseline and then improve from there.
- Full memory-bandwidth saturation should not be used as the single-stream success metric for this workload.
