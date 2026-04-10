# Optimization 5: Local Metal Decode on M4 for Qwen3.5-35B

## Current State (2026-04-09)

Target machine: local Apple Silicon M4 Max.

Target model: `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`

This is a local Metal decode effort, not a Vulkan/RDNA effort.

The current checked-in local baseline from `docs/METAL_PERFORMANCE_PLAN.md` is:

- ZINC decode: about `38.11 tok/s`
- ZINC decode time: about `26.2 ms/tok`

The local llama.cpp reference on the same machine and model is:

- llama.cpp decode: about `52.83 tok/s`
- llama.cpp prompt throughput: about `109.8-111.6 tok/s`

So the current local decode gap is real but not absurd:

- decode gap: about `1.39x`
- absolute decode gap: about `14.7 tok/s`

This is close enough that random churn is a waste. The next wins need to be measured and specific.

## Goal

Improve local Metal decode throughput on the 35B model without losing correctness.

Practical milestones:

- Phase 1 target: `>= 45 tok/s`
- Phase 2 target: `>= 50 tok/s`
- Stretch target: `>= 53 tok/s` local llama.cpp parity

The first serious win is getting into the mid-40s consistently.

## Benchmark Contract

This effort is scored on local Metal decode tok/s for the 35B model.

Primary benchmark:

```bash
zig build bench-metal -- \
  -m /Users/zolotukhin/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --warmup 1 --runs 3 -n 128
```

Keep rules:

1. Use the 3-run average or median, not a single lucky sample.
2. Output still has to start with `Paris` on the short correctness prompt.
3. Do not keep regressions just because a microbench looked better.

Useful supporting measurements:

```bash
zig build bench-metal -- \
  -m /Users/zolotukhin/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --warmup 0 --runs 1 -n 32 --profile

zig build bench-metal-shapes -- \
  -m /Users/zolotukhin/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

The local Metal benchmark path already uses the per-GPU process lock. Do not run overlapping benchmarks on this machine.

## What The Current Evidence Already Says

From `docs/METAL_PERFORMANCE_PLAN.md`, several ideas have already been tested locally and should not be the first thing to retry:

- `--no-repack` was flat
- broad threadgroup override sweeps were neutral
- generic `q8_0` launch overrides did not beat the baseline
- shared-expert dual-`q8_0` reuse was not kept as a whole-model win
- `q8_0` KV on Metal was effectively flat in the tested configuration

That means the next win is unlikely to come from another blind launch sweep.

## Working Hypothesis

The remaining local Metal gap is mostly in one of these buckets:

1. The hottest `q8_0` decode kernels are still weaker than llama.cpp on the real 35B shapes.
2. The runtime still pays too much per-token encoder / barrier / submission overhead.
3. The local working set is larger or less residency-friendly than it needs to be.

Treat every cycle as an attempt to confirm or reject one of those hypotheses.

## Execution Order

### Step 1: Keep the benchmark honest

Files:

- `loops/implement_metal.ts`
- `benchmarks/metal_inference.zig`
- `docs/METAL_PERFORMANCE_PLAN.md`

Requirements:

- benchmark the same local 35B model each cycle
- keep the 3-run measurement discipline
- use the profile output when a cycle stalls

If the benchmark is noisy or contaminated, all later wins are fake.

### Step 2: Trust exact-shape evidence over broad folklore

Files:

- `benchmarks/metal_inference.zig`
- `src/compute/forward_metal.zig`
- `src/shaders/metal/dmmv_q8_0*.metal`

The real local hot shapes already matter more than generic theory:

- LM head `248320 x 2048`
- SSM qkv `8192 x 2048`
- SSM gate `4096 x 2048`
- SSM out `2048 x 4096`

Do not roll out a new kernel variant globally unless the exact-shape bench says it wins first.

### Step 3: Focus on the hottest real decode-side kernels

Files:

- `src/shaders/metal/dmmv_q8_0.metal`
- `src/shaders/metal/dmmv_q8_0_dual.metal`
- `src/shaders/metal/dmmv_q8_0_k2048.metal`
- `src/compute/dmmv.zig`
- `src/compute/forward_metal.zig`

Likely work:

- shape-specific fast paths only where the microbench proves it
- better staging or reduction structure for the hot `q8_0` shapes
- selective kernel routing, not broad replacement

Success criterion:

- exact-shape win first
- whole-model decode win second

### Step 4: Reduce token-tail Metal runtime waste

Files:

- `src/compute/forward_metal.zig`
- `src/metal/command.zig`
- `src/metal/shim.m`

Look for:

- unnecessary `commitAndWait()` calls
- avoidable encoder recreation after barriers
- barriers inserted between dispatches that do not actually need them
- expensive pipeline lookup or buffer bookkeeping on the hot path

This is only worth doing when the profile says it matters. Do not refactor command flow blindly.

### Step 5: Keep memory footprint disciplined

Files:

- `src/compute/forward_metal.zig`
- `src/model/loader_metal.zig`

The local reference path is more conservative about active decode memory.

High-value checks:

- allocate K/V only for real attention layers
- avoid keeping decode-resident state for layers that do not use it
- stay mindful of working-set pressure on unified memory

This is a real decode concern on Apple Silicon, not just a cosmetic cleanup.

### Step 6: Only after the main decode path improves, remove tail work

Files:

- `src/compute/forward_metal.zig`
- `src/shaders/metal/argmax*.metal`

Examples:

- GPU-side greedy argmax
- less logits readback
- smaller CPU-side tail work

These are second-order wins. Do not chase them before the hot kernels and command path.

## What Not To Chase First

- another generic threadgroup sweep with no shape-level evidence
- repack work
- prefix cache / APC style ideas
- sampling changes
- broad infrastructure churn across unrelated files

The local gap is too small for random large rewrites and too large for pure tail cleanup.

## Success Criteria

This effort is succeeding when:

- local 35B decode moves clearly above `38.11 tok/s`
- accepted changes stay coherent
- the profile output gets simpler, not noisier
- the remaining gap to llama.cpp is attributable to a small number of named hotspots

## Likely Files

- `loops/implement_metal.ts`
- `benchmarks/metal_inference.zig`
- `src/compute/forward_metal.zig`
- `src/compute/dmmv.zig`
- `src/metal/command.zig`
- `src/metal/shim.m`
- `src/shaders/metal/dmmv_q8_0.metal`
- `src/shaders/metal/dmmv_q8_0_dual.metal`
- `src/shaders/metal/dmmv_q8_0_k2048.metal`
