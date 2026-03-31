# Decode Throughput Plan

Date: 2026-03-30
Model: `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`
Target GPU: RDNA4 test node (`AMD Radeon AI PRO R9700`, `576 GB/s`, `64 CUs`)

## Goal

Hit a stable `50 tok/s` plain decode path on the 35B model on a single R9700 without breaking correctness.

That is the right next target for the current tree:

1. `33.58 tok/s` is already a healthy bring-up baseline
2. `50 tok/s` is a real step up in usability
3. `50 tok/s` still does not require full llama.cpp parity
4. `50 tok/s` is achievable without pretending a single stream should saturate `576 GB/s`

`50 tok/s` means `20.0 ms/tok`.
The current clean baseline is `29.8 ms/tok`.
So the practical problem is: remove about `9.8 ms` from the token path.

## Current Baseline

Measured on March 30, 2026 on the RDNA4 test node from the current checkout synced to `/root/zinc`.

- clean `ReleaseFast` CLI run: `33.58 tok/s`
- raw `/v1/completions`: `33.55 tok/s`
- raw `/v1/completions` at `concurrency=4`: `33.98 tok/s` aggregate
- 3-prompt reasoning chat matrix: `24.94-28.56 tok/s` (about `26.92 tok/s` average)
- modeled decode bandwidth at `33.58 tok/s`: `112.5 GB/s`, about `19.5%` of `576 GB/s`

Clean CLI command shape:

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

## Latest Profiling Snapshot

Measured on March 30, 2026 on the same clean RDNA4 node after adding the deeper per-phase profiler.

Short profiled run:

- `Generated 16 tokens in 582.0 ms`
- `27.49 tok/s` in profile mode
- `37.21 ms` average GPU token time
- `0.58 ms` average CPU record time
- `38.17 ms` average submit-and-wait time
- `1022` descriptor allocations and `1022` descriptor writes per token
- `0` CPU fallbacks on the benchmark path

Coarse GPU buckets:

- attention: `2.38 ms`
- SSM total: `10.58 ms`
- routed MoE total: `14.12 ms`
- shared expert total: `7.65 ms`
- final tail: `0.92 ms`

SSM breakdown:

- projections: `1.69 ms`
- conv1d: `0.34 ms`
- delta-net update: `7.71 ms`
- gated norm: `0.31 ms`
- out projection + residual: `0.87 ms`

Routed MoE breakdown:

- router projection: `7.58 ms`
- softmax top-k: `2.79 ms`
- gate + up expert projections: `2.13 ms`
- SwiGLU: `0.36 ms`
- down projection: `1.53 ms`
- weighted accumulate: `0.33 ms`

Shared expert breakdown:

- gate + up projections: `6.75 ms`
- SwiGLU: `0.35 ms`
- down projection: `0.52 ms`
- gate accumulate: `0.36 ms`

Important tensor-type facts from that same run:

- SSM projections and `ssm_out` are `q8_0`
- shared expert gate, up, and down projections are `q8_0`
- routed MoE gate and up are `q4_k`
- routed MoE down is `q5_k`

That changes the optimization order a lot. The node is not blocked on CPU recording anymore. It is blocked on:

1. `q8_0` small and medium DMMVs
2. `ssm_delta_net`
3. second-order MoE work like `softmax_topk`

## What 50 tok/s Does And Does Not Mean

`50 tok/s` would model to about `167.5 GB/s` on the current decode-byte estimate.
That is only about `29%` of `576 GB/s`.

So this target does not require:

- matching the current `107 tok/s` llama.cpp baseline
- saturating DRAM bandwidth
- fixing the chat/reasoning path first
- adding concurrency or continuous batching first

It does require:

- a clean fast path with no hidden debug or fallback work
- a better split of CPU token-recording time versus GPU execution time
- much lower host-side setup churn inside the token loop
- better efficiency on the medium and small decode shapes that dominate after correctness

## What The Current Code Suggests

The fast path is no longer the old "151 waits per token" architecture. The biggest architectural bring-up work is already done:

- GPU SSM path exists
- GPU router exists
- shared expert gate has a GPU path
- the token stays in one command buffer through final norm, LM head, and GPU argmax
- greedy sampling only reads back a 4-byte token id on the fast path

That is why the tree is already at `33.58 tok/s`.

The next gap to `50 tok/s` looks different:

### 1. CPU token recording is no longer the main blocker

The command buffer is still re-recorded every token and descriptor churn is still high.
But the new measurements show:

- CPU record is only about `0.58 ms/token`
- GPU execution is about `37.21 ms/token` in profile mode
- no CPU fallback path is firing

So descriptor reuse still matters, but it is no longer the first `50 tok/s` blocker.
It is a cleanup pass, not the next big win.

### 2. Profiling is no longer the blind spot

The deeper profiler answered the important questions:

- CPU record time is sub-millisecond
- fence wait is basically GPU time plus about `0.95 ms` of overhang
- the hot buckets are SSM, routed MoE, and shared expert
- inside those buckets, the real hotspots are now visible

That means the next `9.8 ms` should be chased mostly in kernels, not in host plumbing.

### 3. The remaining gains are concentrated in a few kernels

The RDNA4 tuning notes already show that on the baseline stack:

- raw Vulkan dispatch overhead is tiny
- large matmuls can get close to bandwidth-optimal
- medium and small decode shapes fall off much faster

That matches the current ZINC story. The likely problem is not one giant tail op. It is a mix of:

- `q8_0` decode DMMVs on small and medium shapes
- `ssm_delta_net`
- some second-order routed-MoE work
- possibly conservative barriers after the real hot kernels are better

## Profiling Questions We Already Answered

The current tree can now answer these cleanly:

1. CPU recording is about `0.58 ms/token`
2. actual GPU execution is about `37.21 ms/token` in profile mode
3. submit-to-completion adds only about `0.95 ms` over the measured GPU work
4. the coarse hot buckets are:
   - routed MoE: `14.12 ms`
   - SSM: `10.58 ms`
   - shared expert: `7.65 ms`
5. the fast path still does about `1022` descriptor allocs and `1022` descriptor writes per token
6. no CPU fallback branches fire on the benchmark node

The blind spot is gone. The next work should use these answers, not repeat the same measurement phase.

## Immediate Execution Order

### Phase 1. Keep the low-perturbation phase profiler and use it as the gate

Objective:

- split token latency into CPU record, GPU execute, and wait/readback
- attribute GPU time to major phases, not just whole-token total
- keep profiler overhead low enough to trust

Concrete work:

1. Time token recording on the CPU:
   - start timer before `decode_cmd.begin()`
   - stop after `decode_cmd.end()`
2. Time submit-to-completion separately:
   - start before `submitAndWait`
   - stop after it returns
3. Add coarse GPU timestamps around major phases:
   - embed upload
   - attention layers total
   - SSM layers total
   - MoE routed expert block total
   - shared expert total
   - final norm + LM head + argmax + readback
4. Count per token:
   - descriptor set allocations
   - descriptor writes
   - compute barriers
   - submits
   - fallback-path activations
5. Keep `--profile` overhead under `5%` versus clean mode
6. Treat any profile mode above that overhead as diagnostic-only, not leaderboard data

Current output already gives:

- an honest CPU versus GPU split
- a ranked list of the hot buckets
- subphase timing inside SSM, routed MoE, and shared expert
- startup logs for the active GPU fast paths and tensor types

### Phase 2. Attack the hot kernels in this order

Objective:

- remove the biggest real GPU costs first

Priority order from the latest measurements:

1. `q8_0` router and shared-expert projection path
2. `ssm_delta_net`
3. only then smaller MoE pieces like `softmax_topk`

Concrete work:

1. Build a microbenchmark path for these exact shapes:
   - router projection: `256 x 2048`, `q8_0`
   - shared expert gate/up projections: the model's shared-expert `feed_forward_length x 2048`, `q8_0`
   - SSM output projection: `2048 x 4096`, `q8_0`
2. Capture `RADV_DEBUG=shaderstats` for `dmmv_q8_0.spv` and `ssm_delta_net.spv`
3. Tune `q8_0` with real shader-stat feedback, not with generic LDS tricks
4. Tune `ssm_delta_net` around state-update math, not `conv1d`

Observed dead ends already worth avoiding:

- staging the whole `x` vector in LDS for `q8_0` regressed badly
- forcing a different `q8_0` inner-loop reuse pattern also regressed
- removing the full-softmax pass from `softmax_topk` was effectively noise on this stack

### Phase 3. Eliminate host-side descriptor churn

Objective:

- stop rebuilding descriptor state that is effectively static across tokens

Concrete work:

1. Preallocate persistent descriptor sets for stable bindings at engine init:
   - layer norms
   - attention projections
   - SSM blocks
   - MoE gate/up/down dispatches
   - final norm
   - LM head
2. Reuse descriptor sets across tokens wherever the buffers are stable and only push constants or byte offsets change
3. Keep a small dynamic ring only for the cases that truly need token-local descriptor state
4. Measure CPU record time before and after this change

Why this phase is high priority:

- descriptor allocation and `vkUpdateDescriptorSets` happen before the GPU starts
- that makes them direct token-latency cost, not hidden background cost

Target result:

- materially lower CPU record time per token
- fewer shared-pool resets and fewer hot-path descriptor writes

### Phase 4. Move from re-recording every token toward replay

Objective:

- stop paying full command recording cost every token

Concrete work:

1. After descriptor state is mostly static, pre-record the decode skeleton:
   - either one static primary command buffer
   - or secondary command buffers per layer family / cluster
2. Keep only token-varying pieces dynamic:
   - embedding upload
   - position-dependent push constants
   - final token-id readback
3. Benchmark CPU record time again after partial replay

Why this matters:

- even if raw Vulkan submit overhead is tiny, CPU recording still serializes the token before GPU execution begins

Target result:

- CPU token-recording cost pushed down toward sub-millisecond territory

### Phase 5. Benchmark the real hot decode shapes

Objective:

- tune the kernels that actually gate the current `33.58 -> 50 tok/s` jump

Use these shape buckets first:

- `248320 x 2048` LM head
- `8192 x 2048`
- `4096 x 2048`
- `512 x 2048`
- the batched MoE expert path as it actually runs in decode

Concrete work:

1. Use the existing bandwidth benchmark patterns as the reference shape list
2. Compare ZINC kernels against the RDNA4 tuning targets:
   - vocab output near `90%`
   - `8192 x 2048` near `80%`
   - `4096 x 2048` near `60%`
3. Capture shader stats on the benchmark node:
   - `RADV_DEBUG=shaderstats`
4. Focus on the kernels that are well below target, especially:
   - `q8_0` decode DMMV
   - `ssm_delta_net`
   - any medium-shape path with bad VGPR or LDS pressure

Why this phase comes after profiling and descriptor work:

- if CPU recording is still eating several milliseconds, kernel wins will be masked
- once host overhead is lower, kernel work becomes much easier to measure honestly

Target result:

- the next `3-5 ms` comes from real decode-kernel improvements instead of guesswork

### Phase 6. Audit barriers and force the benchmark path to stay pure

Objective:

- make sure throughput runs are measuring the intended fast path

Concrete work:

1. Log once at startup whether these benchmark-critical GPU paths are active:
   - GPU SSM
   - GPU router
   - GPU shared expert gate
   - GPU argmax
2. Fail or clearly warn in benchmark mode if any of those fall back to CPU
3. Audit every `computeBarrier()` on the fast path and remove the ones that are stronger than needed
4. Keep full logits readback out of throughput runs unless sampling mode requires it

Target result:

- no accidental fallback branches on benchmark numbers
- fewer unnecessary global barriers between small decode ops

### Phase 7. Only after raw decode is near 50 tok/s

These are important, but they are not the next blocker for the single-stream 35B target:

1. close the `chat` versus raw decode gap
2. improve aggregate throughput with concurrency
3. chase higher overall bandwidth utilization through batching

Those should start after the raw plain-decode path is close to `50 tok/s`, not before.

## Practical Runs To Do Next

### 1. Reconfirm the clean baseline

```bash
source .env
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "cd /root/zinc && \
  zig build -Doptimize=ReleaseFast && \
  RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
    -m /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    --prompt 'The capital of France is' --max-tokens 64"
```

### 2. Run the deeper phase profiler on a short decode

```bash
source .env
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "cd /root/zinc && \
  zig build -Doptimize=ReleaseFast && \
  RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
    -m /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    --prompt 'The capital of France is' --max-tokens 16 --profile"
```

### 3. Capture shader stats for the hot kernels

```bash
source .env
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "cd /root/zinc && \
  RADV_DEBUG=shaderstats RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
    -m /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    --prompt 'The capital of France is' --max-tokens 4"
```

## Success Criteria

### Short term

- clean `ReleaseFast` 35B decode remains reproducibly above `33 tok/s`
- new profiler overhead stays within a defensible diagnostic range
- every throughput run reports whether a CPU fallback path was used
- the profiler keeps identifying the same top buckets reproducibly
- the next accepted kernel change reduces one of:
  - routed MoE router time
  - shared expert projection time
  - `ssm_delta_net` time

### Medium term

- CPU token-recording time drops materially after descriptor reuse
- no benchmark-critical path depends on mid-token CPU fallback
- at least one hot shape family is benchmarked against RDNA4 reference utilization

### Primary target

- stable `>= 50 tok/s` plain decode on the 35B model on the clean R9700 node
- equivalent to `<= 20.0 ms/tok`
- modeled decode bandwidth at or above about `167 GB/s`

## Notes

- The old "stuck at `7 tok/s`" story was partly a tooling problem and partly an architecture problem. That is no longer the current state.
- The current fast path is healthy enough that the next wins should be measured as milliseconds removed from `29.8 ms/tok`, not as vague utilization improvements.
- The smaller `Qwen3.5-2B-Q4_K_M` still being slower than the 35B MoE model is a useful clue: this codebase is now dominated by decode-kernel regime and control overhead, not just raw model size.
- Full single-stream memory-bandwidth saturation should not be treated as the immediate success metric for this target.
