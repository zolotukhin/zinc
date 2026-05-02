# Effort 12 - Metal Gemma 4 31B on M4: dense decode + prefill

## Objective

Make Gemma 4 31B usable on local Apple Silicon M4 through the Metal backend.
Coherence is already there: chat mode emits "The capital of France is Paris."
on the canonical prompt. Throughput is not. The 14-token chat prefill takes
72 seconds and decode reads at 0.28 tok/s. The model fits in 64 GB UMA at
Q4_K_M, so this is a kernel/dispatch problem, not a memory problem.

Primary model for this effort:

- `gemma4-31b-q4k-m` from the managed cache (17.5 GB on disk).
- Local machine: Apple GPU family Apple9 / M4, 64 GB unified memory, 546
  GB/s peak bandwidth.
- Prompt mode: chat template, not raw completion.

Run the loop with:

```bash
ZINC_MODEL_ID=gemma4-31b-q4k-m \
ZINC_PROMPT_MODE=chat \
ZINC_TEST_PROMPT="What is the capital of France?" \
ZINC_MAX_TOKENS=12 \
ZINC_TARGET_TOK_PER_SEC=20 \
ZINC_STOP_ON_TARGET=0 \
ZINC_BENCHMARK_RUNS=3 \
ZINC_PROFILE_EVERY=1 \
ZINC_BUILD_OPTIMIZE=ReleaseFast \
ZINC_TEST_TIMEOUT_MS=300000 \
ZINC_RUN_TIMEOUT_MS=1800000 \
ZINC_CODEX_REASONING_EFFORT=xhigh \
bun loops/implement_metal.ts --resume --effort 12 --agent codex --model gpt-5.5 --cycles 100
```

`--agent claude` is also fine; the doc is written for either agent. Use
`ZINC_BENCHMARK_RUNS=3` while every run is multi-minute. Step up to 5
samples once decode is past 5 tok/s.

`ZINC_RUN_TIMEOUT_MS=1800000` is 30 minutes, sized for the current 70-second
prefill and to leave headroom for kernel-load slowness early in the effort.
Tighten this back toward 600000 once prefill is under 30 seconds.

Important harness detail:

- `implement_metal.ts` must build the verifier binary with
  `zig build -Doptimize=ReleaseFast`.
- If the loop says `Building (zig build)` or the verifier measures a binary
  produced by plain `zig build`, stop and fix the harness before optimizing.
- Agent-side `--profile` numbers are not accepted unless the official loop
  verifier was built with the same optimize mode.

## Current baseline

Loader output on the canonical chat prompt:

```text
Architecture: gemma4 | 60 layers | 32 heads (16 KV) | dim 5376 | vocab 262144
File size: 17474 MB
Tensors: 833
Metal context trimmed from 262144 to 8450 tokens to fit current UMA budget
```

Post-cycle-24 measurement (2026-05-01):

```text
Prefill: 20 tokens in 2061.6 ms (9.7 tok/s)
Decode: 0.29 tok/s (3501.0 ms/tok) on the 7-token chat completion
Output: "The capital of France is Paris." (coherent)
Best accepted across cycles 1-24: 0.30 tok/s (cycle 18)
```

Initial baseline (pre-cycle-1, on Effort-11 cycle-49 runtime):

```text
Prefill: 14 tokens in 72.4 s (0.19 tok/s)
Decode: 0.28 tok/s
```

Diagnosis:

- The model is **dense, not MoE**. Effort 11's Gemma-12B foundations
  optimized the MoE-routed path (`canUseGpuRoutedBatchedMoe`,
  `dmmv_q5_1_moe`, `moe_route_pack`, etc.). The 31B does not enter that
  code path. It either falls through to a per-token CPU dense FFN or to a
  generic Metal DMMV path that nobody has tuned for the 31B's larger
  shapes.
- 60 layers is 2x the 12B layer count. Even at the same per-layer cost,
  total cost doubles.
- `dim` is 5376 vs 12B's 2816 (1.91x wider). Hot DMMV K dimensions are
  not 2816 anymore; the cycle-49 K-cap and pair-Q8 routing decisions do
  not transfer.
- LM head shape is `M=262144 K=5376` vs 12B's `M=262144 K=2816`. The
  CPU Q8 LM-head fallback Effort 11 spent cycles 7-14 tuning runs into
  ~1.9x more bytes per call.
- 14 tokens / 72.4 s = 0.19 tok/s **prefill**. That is two orders of
  magnitude below where 12B prefill landed at the start of Effort 11
  (which was already ~2.2 tok/s prefill before any Metal work). Every
  decoded token also waits behind a multi-second-per-step path.

Bandwidth ceiling math, updated post-cycle-24:

```text
Per-token byte traffic (cycle-24 profile): Q4_K 98.97 GiB + Q6_K 21.49 GiB
  across 7 tokens = 17.2 GiB/token. This matches the dense-weight working
  set, so the GPU is reading the right amount of data.
Per-token wall time: 3501 ms (commitAndWait, 100% of traced time).
Effective bandwidth: 17.2 GiB / 3.501 s = 4.9 GB/s.
M4 peak bandwidth: 546 GB/s. Achievable ceiling ~480 GB/s.
Effective is ~1% of peak. The kernel is the bottleneck. Dispatch overhead
is not (commits=7 for 7 tokens, already one commit per decode step). CPU
record time is ~5 ms total per step; the remaining 3496 ms is GPU work
running at sub-bandwidth throughput.
```

This is a different bug than the Effort-11 cycle-71-92 measurement
collapse. There the verifier was noisy on stable code; here the verifier
is consistent and the GPU is genuinely slow. Tuning the kernel at the
margin does nothing because the kernel architecture is wrong for Apple9.

## Measurement environment

Cycles 71-145 of Effort 11 demonstrated that long uninterrupted runs on
this M4 produce 12-25 tok/s readings on what should be 37.79 tok/s code.
Verifier sample range routinely exceeds 30% of median when the machine has
been under load for several hours.

Before claiming any speedup or regression on this effort:

1. Run the verifier command 5 times on the accepted-best code at HEAD with
   no agent edit. Drop sample 1 (warmup) and report median + range of
   the next 4.
2. If decode median is below the prior accepted best by more than 15%, the
   environment is degraded. Do not optimize against a moving baseline.
   Reduce contention (close other heavy processes, let the machine cool)
   and re-measure.
3. The harness has variance/dormant guards in the keep/reject path
   (cycle-92 follow-up). Use them: a verifier with range > 30% of median
   is inconclusive, not a regression.

## Reference implementations

The closest production reference for dense Gemma 4 27B/31B:

- `/Users/zolotukhin/Workplace/llama.cpp/src/llama-model.cpp`
  - Look for `LLM_ARCH_GEMMA3` / `LLM_ARCH_GEMMA4` graph builders. The
    27B/31B variants use the same per-layer pre/post norms, per-layer
    RoPE, and SWA stride that the 12B path already handles in ZINC.
- `/Users/zolotukhin/Workplace/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp`
  - `ggml_metal_op_mul_mat`: dispatch decisions for dense matmul on
    Apple. The shape thresholds inside this function are the canonical
    set for Apple9.
- `/Users/zolotukhin/Workplace/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`
  - `kernel_mul_mv_q4_K_f32` and `kernel_mul_mv_q8_0_f32` for the dense
    matvec shapes that dominate decode. Note the per-shape NR0/NSG
    decisions and the row-pairing constants.

Effort 11's `Reference implementations` section still applies for routing
patterns; it just is not load-bearing here because 31B is dense.

## Already landed foundations

These are available on `main` and should not be reimplemented:

- ZINC's Metal backend boots Gemma 4, dispatches per-layer norms, RoPE,
  flash attention, and a dense FFN path. The 31B coherent output proves
  the basic graph works.
- Q8 paired DMMV variants (`dmmv_q8_0_pair`, `dmmv_q8_0_pair_k2816`,
  `dmmv_q8_0_dual`, `dmmv_q8_0_quad`) shipped as part of Effort 11.
  Reuse them where shapes match. Do not retune their threadgroup sizes
  for 31B without bench-metal-shapes evidence on the new shapes.
- `dmmv_q4k.metal`, `dmmv_q4k_dual.metal`, `dmmv_q4k_lmhead.metal`,
  `dmmv_q4k_lmhead_1024.metal`, `dmmv_q4k_k2048.metal` cover the dense
  Q4_K matvec shapes for 12B. The K dimension does not match 31B; new
  variants may be needed.
- CPU Q8 LM-head fallback is parallelized (Effort 11 cycle 8). The same
  worker pool will pick up the 31B LM-head; it just will move 1.9x more
  bytes per call.
- Batched prefill exists for Gemma 12B behind `ZINC_GEMMA_BATCHED_PREFILL`.
  It is unclear whether the 31B falls into the same batched path or
  takes a per-token loop; verify in Step 0.
- Effort 11's measurement infrastructure (`bench-metal-shapes` cases for
  `attn_q`, `attn_k`, `attn_v`, `attn_output`, `lm_head`, `shared_gate`,
  `shared_up`, `shared_down`, `moe_down`, `moe_gate_up`, `shared_block`,
  `lm_head_argmax`) lives in `benchmarks/metal_q8_shapes.zig`. Use these.
  Add 31B-shape cases instead of editing kernels first.

## Definition of "decent speed"

Minimum acceptable milestone:

- Coherent Paris answer in chat mode at default settings.
- `zig build test` passes.
- Official ReleaseFast verifier reaches at least `5 tok/s` decode.
- 14-token chat prefill drops below 30 seconds.

Target milestone:

- Official ReleaseFast verifier reaches at least `15 tok/s` decode.
- 14-token chat prefill drops below 10 seconds.
- Profile shows named remaining bottlenecks rather than generic
  fallback-dense or per-token CPU paths.

Stretch milestone:

- Decode reaches `25 tok/s`, within ~85% of the M4 bandwidth ceiling for
  a dense Q4_K_M 31B.
- Prefill drops below 5 seconds for the 14-token chat prompt (i.e.
  prefill at >=2.8 tok/s, ~10x today).

Hard ceiling:

- Decode is bandwidth-bound at ~31 tok/s on this card. Do not chase
  numbers above 30 unless prefill is also dramatically improved or the
  bandwidth math changes (e.g. quant tier, KV layout).

## What cycles 1-40 taught us (and why kernel tuning is done)

Updated 2026-05-01 after cycle 40. `bench-metal-shapes --case dense_q4k_5376`
now gives ground truth: the dense Gemma 31B Q4_K/Q6_K kernels are not the
bottleneck.

```text
LM head        Q4_K M=262144 K=5376 → 491 GB/s
attn_q         Q4_K M=8192   K=5376 → 594 GB/s
attn_k         Q4_K M=4096   K=5376 → 561 GB/s
attn_v         Q6_K M=4096   K=5376 → 453 GB/s
attn_out       Q4_K M=5376   K=8192 → 629 GB/s
ffn_gate       Q4_K M=21504  K=5376 → 504 GB/s
ffn_up         Q4_K M=21504  K=5376 → 490 GB/s
ffn_down       Q4_K M=5376   K=21504 → 499 GB/s
```

**All kernels run at or above the M4's ~480 GB/s effective ceiling.**

Sum-of-kernels per layer ≈ 1.0 ms. Times 60 layers = 60 ms per token if
the dispatches were perfectly pipelined. Real cycle-40 measurement is
3460 ms per token. The 57x gap is between dispatches, not inside them.

The dispatch chain alone needs ~3.4 s per token for ~600 small dispatches.
That points at one or more of:

1. Encoder switch / barrier overhead (each `mtl_barrier` creates a new
   compute encoder; on Apple this is ~10-30 us per switch, but the math
   here implies something much heavier).
2. Pipeline stalls from dependent dispatches that can't overlap.
3. Buffer rebinding cost per dispatch.
4. GPU clock not ramping to full speed when individual dispatches are
   ~50 us — power management leaving the card idle most of the wall
   clock.
5. Some ZINC-specific control-flow round-trip per dispatch.

**Stop tuning kernels.** Cycles 41+ must measure dispatch cadence and
GPU-active vs GPU-idle time on the real run, then attack the gap. Adding
another shader variant is wasted unless it removes a barrier or fuses
two dispatches into one.

## What cycles 1-24 taught us

23 of 24 cycles were "kept-within-noise" (only cycle 5 broke coherence
and was reverted). Best decode moved 0.21 → 0.30 tok/s, then drifted
back to 0.29. Total signal: ~0.08 tok/s, well inside the harness's 0.3
tok/s noise band — i.e. nothing actually improved.

What was tried, and why it failed:

- **Cycles 2, 7, 11, 14, 15, 16, 17, 18, 21, 22 retuned `dmmv_q4k_k5376.metal`.**
  The kernel was created in cycle 2 with `TG_SIZE=512`, 16 rows per
  threadgroup, 1 simdgroup per row, 21 KiB threadgroup-cached input
  vector. Subsequent cycles tweaked unrolls, dispatch routing,
  simdgroup-matrix variants, dual gate/up. None of these moved decode
  past 0.30 tok/s because the underlying architecture is wrong for
  Apple9 — TGM bank contention across 16 simdgroups and per-row
  simdgroup at low occupancy keep effective bandwidth at ~5 GB/s.
- **Cycles 8, 9, 23, 24 chased command-buffer chunking and async submit.**
  CPU record time is 5 ms per step; chunking that further does
  nothing when GPU wait is 3500 ms.
- **Cycle 12 copied tensors out of mmap into Metal-owned buffers.**
  This is on Effort 11's dead-end list (cycle 63 - mmap is fine).
- **Cycle 13 enabled batched prefill default-on without validation.**
  This is on Effort 11's dead-end list (cycle 69 - requires logits
  parity gate first).
- **Cycle 1 edited `loops/implement_metal.ts`.** The harness rules
  forbid that. The agent must not touch the harness.

The harness's keep-within-noise band of 0.3 tok/s is too wide at this
scale: at a 0.21 tok/s baseline, every diff under +0.6 tok/s is
"kept-within-noise". 23 of 24 cycles got accepted that way. The
in-flight pre-cycle empty commits accumulated even though zero
real progress happened.

## Execution order

### Step 0 - Verify the dense path is on GPU

Do not start by changing kernels. First confirm where the 72-second
prefill and 0.28 tok/s decode actually go.

1. Build `zig build -Doptimize=ReleaseFast`.
2. Run the chat prompt with `--profile` and a small `-n` to get a
   request profile. The expected output identifies steps, commits,
   record breakdown, and per-quant byte traffic.
3. Confirm:
   - `cmds` and `commits` per token. If commits per token are large
     (hundreds), the per-layer command-buffer chain is missing.
   - `record breakdown`: which phase dominates `commitAndWait`? For
     12B this is "final" (CPU LM-head) and "gpu-moe". For 31B the
     "fallback" and "dense" buckets are the ones to watch.
   - `dmmv bytes/request`: what the per-step byte budget actually is.
     Compare against 17.5 GB working set; large miss means weights
     are being read more than once per token.
4. Run the same prompt without `--profile`. The numbers should be the
   same within noise; if they differ by more than 20%, the profile is
   adding overhead that is hiding the real cost.

Acceptance:

- A profile snapshot is captured under `loops/efforts/` or pasted into
  the next cycle's self-analysis.
- The dominant phase is named (not "agent made changes").
- The next cycle is targeted at that phase, not at a guess.

### Step 1 [DONE 2026-05-01] - K=5376 kernel removed

The post-cycle-24 K=5376 specialization (`dmmv_q4k_k5376.metal`) was
deleted in commit `22bbc75`. K=5376 traffic now falls through to the
base `dmmv_q4k.metal` (llama.cpp port, NSG=2, NR0=2, 64 threads).
Microbench confirms the base kernel hits 491-629 GB/s on every dense
31B shape. **Kernel tuning is closed.** Step 2 is now the live problem.

Historical context for this step:

`dmmv_q4k_k5376.metal` (created cycle 2) was a 512-thread/16-row/TGM-
cached variant that ran at ~5 GB/s effective on Apple9. llama.cpp's
`kernel_mul_mv_q4_K_f32` on Apple9 hits ~440 GB/s on the same K=5376
shapes. Stop tuning the existing kernel; rebuild from the llama.cpp
reference shape.

(Original Step 1 detail kept below for reference; do not re-execute.)

Required architectural choices (all from llama.cpp):

```
NSG  = 2          # 2 simdgroups per threadgroup
NR0  = 2          # 2 rows per simdgroup → 4 rows per TG total
TG   = 64         # threads per threadgroup
NO TGM input cache. The GPU's L2 already covers this access pattern
on Apple9; the 21 KiB TGM cache in the current kernel adds bank
contention without saving bandwidth.
```

Required steps:

1. Add `bench-metal-shapes --case dense_q4k_5376` covering all 31B
   decode shapes (Q/K/V/O projections, dense FFN gate/up/down, LM head
   if it routes through Q4_K). Report effective GB/s for each shape.
2. **Land a port of llama.cpp `kernel_mul_mv_q4_K_f32` as the default
   path.** Use the same threadgroup/simdgroup/row layout. Do not invent
   a wider variant. ZINC's existing `dmmv_q4k.metal` is already a port
   of this shape for K=2816; extend its dispatch to also handle K=5376
   instead of routing through the broken specialized kernel.
3. After the port lands, delete `dmmv_q4k_k5376.metal` and the runtime
   pipeline reference in `forward_metal.zig`. The kernel was a wrong
   design; it should not stay around as a "fallback".
4. Run `bench-metal-shapes` again to confirm the new path is
   >=300 GB/s on all K=5376 shapes.

Acceptance:

- `bench-metal-shapes` reports >=300 GB/s on all six K=5376 shapes.
- `dmmv_q4k_k5376.metal` is gone.
- Verifier decode improves by at least 5x (target >=1.5 tok/s).
- If the port lands and decode stays under 1.0 tok/s, the bottleneck
  is somewhere else (e.g. attention, KV cache, sampling). Stop. Do
  Step 0 again with an updated profile.

### Step 2 [LIVE] - Find and remove the per-dispatch overhead

This is the active step as of 2026-05-01.

The kernel side is fine. Per-token GPU work is 3460 ms but the sum of
the kernels is ~60 ms. The 57x gap is dispatch overhead.

Static count of barriers in the decode hot path (`src/compute/forward_metal.zig`
lines 2868-3001, dense Gemma path): ~10-15 `cmd.barrier()` per layer
× 60 layers = **600-900 barriers per token**, plus 3 more after the
layer loop. The bench tool dispatches the same kernels 50 times in a
tight loop with no barriers and gets 500 GB/s; the real run forces
strict ordering on every phase boundary.

`cmd.barrier()` calls `memoryBarrierWithScope:MTLBarrierScopeBuffers`
inside a concurrent compute encoder (`shim.m::mtl_begin_command_mode`,
`MTLDispatchTypeConcurrent`). Independent dispatches between barriers
overlap freely; barriers serialize. Almost every barrier in the
current dense decode is data-dependency-required (RMS → Q/K/V → RoPE
→ KV write → FA → O → residual → RMS → gate/up → SwiGLU → down →
post-norm → scale_acc), so removing barriers without fusion will
break correctness.

The lever is **fusion**, not barrier removal:

1. Fuse `scale_acc` (residual add) + next layer's `pre-attn RMS` into
   one kernel. That kills one barrier per layer = 60 barriers per
   token gone.
2. Fuse `RMS norm` + (`Q` projection) into one kernel that reads the
   hidden vector once and writes Q directly. K and V can run
   concurrently from the same norm output.
3. Fuse `RoPE` + `KV cache write` into one kernel for the K side
   (V already does not need RoPE).
4. Fuse `O projection` + `residual add into hidden` so the post-attn
   path lands the residual in one dispatch.
5. Fuse `gate` + `up` + `GeGLU` into one kernel. Already partially
   present (`dmmv_q4k_dense_gate_up_geglu.metal` from cycle 36).
   Verify it actually replaces three dispatches with one in the
   default decode path; if not, wire it in.
6. Fuse `down` + `post-ffn RMS` + `scale_acc` into one kernel.

Each successful fusion removes one or more barriers. Six fusions ×
60 layers = 360-540 fewer barriers per token. If a barrier costs even
4 ms, that is 1.4-2.2 s/token recovered.

Required measurement first (single cycle, no kernel change):

1. Add a counter in `forward_metal.zig` decode that, on the first
   decoded token, reports:
   - total `cmd.barrier()` calls
   - total `cmd.dispatchV2` (and v2_tgmem) calls
   - count of `mtl_begin_command` / `commit_and_wait` per token
   - per-phase dispatch count (attn, ffn, norm, residual)
2. Print once on cycle entry, then continue with default behavior.
3. Compare the printed barrier count against the math (~600-900) to
   confirm the hypothesis before fusing anything.

After the count is known, each subsequent cycle implements ONE
fusion from the list above and reports:
- before/after `cmd.barrier()` count
- before/after per-token GPU wait ms
- bench-metal-shapes for any new fused shape

Acceptance:

- First cycle on Step 2 lands instrumentation only.
- Each fusion cycle removes a measurable number of barriers and is
  evaluated by the harness against decode tok/s.
- When per-token GPU wait drops below 200 ms or barrier count drops
  below 200 per token, Step 2 is done.

Out of scope on this step:

- Adding more matvec variants. Kernels run at ~500 GB/s; do not
  retune.
- Tensor copy / arena packing experiments. Already done in cycle 40,
  no decode improvement.
- Flag-gated paths. Default-on or do not write.
- `bench-metal-shapes` cases unrelated to a fusion landing the same
  cycle.

### Step 2 [LEGACY] - Identify and fix the dense FFN dispatch

The 31B dense FFN gate/up/down shapes are larger than 12B's MoE expert
shapes. If the current code path falls back to a CPU loop for any
projection, that is the dominant cost.

1. From the Step 0 profile, locate the dense FFN dispatch site in
   `forward_metal.zig`. If it is missing (model uses CPU), trace where
   the architecture branch chose CPU.
2. Wire dense Q4_K gate/up to a Metal pair dispatch (analogous to
   `dmmv_q4k_dual` for fused gate+up where applicable).
3. Wire dense down to a Q4_K Metal DMMV at `M=hidden K=intermediate`.
4. Keep all post-FFN norms and residuals on GPU.

Acceptance:

- The agent-side profile shows the dense FFN under "layer" or "dense"
  buckets, not "fallback".
- Verifier decode tok/s improves materially (target +5 tok/s over Step
  1's exit point).
- Coherence holds.

### Step 3 - Decide the LM-head path

The 31B LM head is `M=262144 K=5376`. At Q8_0 that is ~1.4 GB read per
token. The 12B's CPU LM-head fallback (Effort 11 cycles 7-8) reads ~750
MB per token; the 31B is ~1.9x heavier.

Do not assume CPU is the right path for 31B without measurement. The
Apple9 raw GPU LM-head DMMV moved ~440 GB/s on 12B; the same kernel on
31B should move similar bandwidth, which translates to ~3.2 ms per
token. The CPU LM-head, parallelized with 16-lane vectors, may beat or
lose to that.

1. Add `bench-metal-shapes --case lm_head_5376` for both the GPU DMMV
   and a CPU reference reading the same buffer.
2. If GPU is faster, wire the LM head to a Metal Q8 matvec and keep
   sampling on GPU as well to avoid a logits readback.
3. If CPU is faster, parallelize and vectorize the existing fallback
   for the larger K.

Acceptance:

- The "final" phase in the profile drops to <100 ms per token.
- Verifier coherence holds. Sampling correctness must be verified
  against a ground-truth ZINC run on the 12B model first; see Effort
  11 cycles 21, 22 for the argmax buffer reuse pattern.

### Step 4 - Prefill

The 14-token chat prefill is 72 seconds. Even per-token decode at the
target rate (15 tok/s = 67 ms/tok) would do 14 tokens in ~0.94 s. The
prefill is doing something dramatically slower than decode.

Two likely failure modes:

- The dense path uses a per-token loop with no batching, so 14 tokens
  cost 14 * decode-step time. If decode-step is ~3.6 s, total is ~50 s
  plus warmup. That matches the observed 72 s.
- The dense path uses a CPU fallback during prefill that decode does
  not hit (e.g. some cold-start tensor materialization).

Order of operations:

1. From the Step 0 profile, identify whether prefill is bound by per-
   token cost or by something one-time. If per-token, jump to (2);
   if one-time, fix that first.
2. Use the existing batched prefill machinery from Effort 11's Step 7.
   The 31B is dense, so the MoE batched-prefill scratch (`moe_route_*`)
   does not apply. The dense batched prefill should:
   - batch hidden states `[N, hidden_dim]` through a single GEMM per
     projection,
   - run flash attention in batched mode,
   - run dense FFN gate/up/down on the batched activations,
   - keep last-token logits computation off the batched path or fold
     it into a final per-token step.
3. Validate logits parity on a 14-token prompt before flipping the
   default. The harness's `ZINC_BATCHED_PREFILL=validate` plumbing from
   Effort 11 cycle 92 still applies.

Acceptance:

- 14-token chat prefill drops below 10 seconds.
- Logits parity holds (last-token max abs diff < 1e-3 vs per-token).
- Decode is unchanged.

### Step 5 - Validate against bigger prompts

Once the chat prompt is fast, the effort isn't done. The 12B effort
optimized exclusively for the 14-token "What is the capital of France?"
prompt and acquired blind spots in longer contexts.

1. Add a second loop prompt: a 256-token chat with a non-trivial
   question (e.g. summarize a paragraph). Run it once per accepted
   cycle, even if it is not the verifier benchmark.
2. Confirm prefill scales sublinearly past 32 tokens (batched prefill
   should beat the per-token path by 2-5x).
3. Confirm decode tok/s does not collapse at 1k context.

Acceptance:

- Both the 14-token and 256-token prompts are coherent.
- Decode tok/s at 1k context is within 70% of decode tok/s at 0
  context.

## Known dead ends - do not repeat

Effort 12 specific (cycles 1-24):

- **`dmmv_q4k_k5376.metal` (TG=512, 16 rows/TG, simdgroup-per-row,
  21 KiB TGM input cache).** Created cycle 2, retuned cycles 7, 11,
  14-18, 21, 22. Runs at ~5 GB/s effective on Apple9. Wrong design.
  Replace with a port of llama.cpp `kernel_mul_mv_q4_K_f32` (NSG=2,
  NR0=2, 64 threads, no TGM input cache) before any further dense
  decode work.
- Command-buffer chunking and async submit (cycles 8, 9, 23, 24).
  Useless when CPU record time is 5 ms and GPU wait is 3500 ms.
  Re-open only after decode is past 5 tok/s.
- Tensor copy out of mmap into Metal-owned buffers (cycle 12). Same
  conclusion as Effort 11 cycle 63: mmap is fine.
- Default-on batched prefill without `ZINC_BATCHED_PREFILL=validate`
  logits parity (cycle 13). Same conclusion as Effort 11 cycle 69.
- Editing `loops/implement_metal.ts` from inside an agent cycle
  (cycle 1). The harness rules forbid this. Any cycle that touches
  `loops/` will be reverted at minimum and may break the harness's
  rule-enforcement logic.

Inherited from Effort 11. Read Effort 11's full dead-end list before
proposing any kernel-level work; the highlights for 31B are:

- Q5_1 MoE down threadgroup retunes (12B-specific). 31B is dense; the
  Q5_1 MoE shaders should not be touched.
- 256-thread paired Q8 K/V override regressed 12B cycle 46. The 31B
  K/V shapes are different (32 heads / 16 KV vs 16/8); a fresh paired
  Q8 path may be appropriate, but it must come with `bench-metal-shapes`
  evidence at K=5376.
- Removing the Gemma shared gate/up 256-thread special case regressed
  12B cycle 67. 31B is dense and does not have a "shared expert", but
  the Apple9 512-thread vs 256-thread tradeoff is the same axis.
  Confirm with bench-metal-shapes before broadening.
- Default-on Gemma batched prefill without validation-on logits parity
  regressed 12B cycle 69. Same gate applies for 31B Step 4.
- Measurement-only / opt-in flag cycles do not move the verifier and
  are wasted slots. The harness silently revert-skips them now, but
  they still consume the slot. Bundle measurement with a default-on
  production edit, or skip it.
- "No code change / studied references" cycles must end in a kept
  measurement coverage change or a concrete no-code conclusion in
  this effort doc; otherwise they are pure burn.

## Measurement gates

Every kept change must include:

```bash
zig build test
zig build -Doptimize=ReleaseFast
./zig-out/bin/zinc --model-id gemma4-31b-q4k-m \
  --prompt "What is the capital of France?" --chat -n 12 --profile
```

If a change touches generic Metal kernels, also run a Gemma-12B smoke
to make sure Effort 11's gains did not regress:

```bash
./zig-out/bin/zinc --model-id gemma4-12b-q4k-m \
  --prompt "What is the capital of France?" --chat -n 12
```

And a Qwen smoke:

```bash
./zig-out/bin/zinc --model-id qwen3-8b-q4k-m \
  --prompt "What is the capital of France?" --chat -n 8
```

Reject a change if:

- It loses Paris coherence on 31B chat.
- It regresses 12B chat decode below the Effort 11 accepted best.
- It moves work from profile-visible GPU time into unprofiled CPU work.
- It improves raw 31B completions but regresses chat mode.
- It is justified only by an agent-side `--profile` speedup while the
  official verifier does not reproduce the gain under the same optimize
  mode.
- It is measurement-only / opt-in / validation-only and does not change
  default decode or prefill behavior on 31B.
- It optimizes from a verifier run whose sample range is wider than 25%
  (looser than 12B's 20% because 31B samples are slower and noisier).

## Files likely to change

- `loops/implement_metal.ts` - harness config only; do not keep changing
  it after Step 0.
- `src/compute/forward_metal.zig` - dense FFN orchestration, prefill
  batching, LM-head path selection.
- `benchmarks/metal_q8_shapes.zig` - new bench cases for K=5376.
- `src/shaders/metal/dmmv_q4k.metal` and friends - shape-specific
  variants for K=5376 if needed.
- New shaders under `src/shaders/metal/` only when a microbench shows
  the existing variants leave bandwidth on the table.

Do **not** edit:

- `src/shaders/metal/*moe*.metal` - 31B is dense; touching MoE shaders
  risks 12B regressions and gives 31B nothing.
- `src/vulkan/` - Vulkan-side perf-effort work owns this directory.

## Expected end state

```text
Gemma 31B chat prompt
  -> batched hidden states [N,H]
  -> Q4_K dense projections (Q/K/V/O) at K=5376 on GPU
  -> batched flash attention
  -> Q4_K dense FFN (gate/up/down) at intermediate dim on GPU
  -> post norms / residuals on GPU
  -> last-token Q8 LM head (GPU or fast CPU, whichever wins the bench)
  -> argmax / sampling close to where logits land
```

For decode the same path runs at `N=1`, but the win is keeping every
phase on the same command buffer and avoiding per-token CPU readbacks.
The bandwidth ceiling on this card is ~31 tok/s for dense 31B at
Q4_K_M; the target is to spend the rest of the budget on closing the
20-30x gap between today's 0.28 tok/s and that ceiling.
