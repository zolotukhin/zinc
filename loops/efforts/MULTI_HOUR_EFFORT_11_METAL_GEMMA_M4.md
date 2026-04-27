# Effort 11 - Metal Gemma 4 on M4: correctness + decent speed

## Objective

Make Gemma 4 usable on local Apple Silicon M4 through the Metal backend.
Correctness is mandatory, but the current path is too slow to be useful:
Gemma 4 chat works only through the chat template and spends nearly all time
in the Gemma MoE fallback path.

Primary model for this effort:

- `gemma4-12b-q4k-m` from the managed cache.
- Local machine: Apple GPU family Apple9 / M4, unified memory.
- Prompt mode: chat template, not raw completion.

Run the loop with:

```bash
ZINC_MODEL_ID=gemma4-12b-q4k-m \
ZINC_PROMPT_MODE=chat \
ZINC_TEST_PROMPT="What is the capital of France?" \
ZINC_MAX_TOKENS=12 \
ZINC_TARGET_TOK_PER_SEC=50 \
ZINC_STOP_ON_TARGET=0 \
ZINC_BENCHMARK_RUNS=3 \
ZINC_PROFILE_EVERY=1 \
ZINC_BUILD_OPTIMIZE=ReleaseFast \
ZINC_TEST_TIMEOUT_MS=300000 \
ZINC_RUN_TIMEOUT_MS=900000 \
ZINC_CODEX_REASONING_EFFORT=xhigh \
bun loops/implement_metal.ts --resume --effort 11 --agent codex --model gpt-5.5 --cycles 100
```

Use `--agent claude` if desired; the effort is written for either agent.
Use `ZINC_BENCHMARK_RUNS=1` only for quick triage. The cycle-23 to cycle-38
region has too much noise for one-sample keeps.

Important harness detail:

- `implement_metal.ts` must build the verifier binary with
  `zig build -Doptimize=ReleaseFast`.
- If the loop says `Building (zig build)` or the verifier measures a binary
  produced by plain `zig build`, stop and fix the harness before optimizing.
- Agent-side `--profile` numbers are not accepted unless the official loop
  verifier was built with the same optimize mode.

## Current baseline

Current post-cycle-38 state:

```text
Best official loop verifier: 15.89 tok/s (cycle 33)
Current official loop verifier: 15.80 tok/s (cycle 38)
Current stall: 17 cycles
Cycle-38 agent ReleaseFast profile: 36.55 tok/s, correct Paris output
Cycle-38 official verifier: 15.80 tok/s, correct Paris output
Post-harness-fix manual ReleaseFast profile: 28.70 tok/s, correct Paris output
Post-harness-fix manual ReleaseFast no-profile run: 25.73 tok/s, correct Paris output
```

Diagnosis:

- The old official verifier path was rebuilding with plain `zig build`.
- Agents were profiling after `zig build -Doptimize=ReleaseFast`.
- That means the loop's keep/revert decision was comparing Debug-ish verifier
  throughput against ReleaseFast agent profiles.
- The next cycle must first establish a new ReleaseFast verifier baseline. Do
  not spend another cycle on speculative kernel retunes until the verifier and
  profile agree within normal run-to-run noise.

Expected first run after the harness fix:

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zinc --model-id gemma4-12b-q4k-m \
  --prompt "What is the capital of France?" --chat -n 12 --profile
```

The output must contain Paris and should be treated as the new baseline only if
the loop verifier reports the same optimize mode and roughly the same decode
range. If the profile is around `25-40 tok/s` but the verifier remains around
`15 tok/s`, fix measurement before changing model kernels.

Measured locally before this effort file was created:

```text
./zig-out/bin/zinc --model-id gemma4-12b-q4k-m \
  --prompt "What is the capital of France?" --chat -n 12 --profile

Output: The capital of France is **Paris**.
Prefill: 20 tokens in 90838.4 ms (0.2 tok/s)
Generated 8 tokens in 39523.7 ms - 0.20 tok/s (4940.5 ms/tok)
Metal profile: cmds=1689 commits=1689
record breakdown: fallback-moe 48714.54 ms, final 5227.43 ms
mix/step: attn 30.0, fallback-moe 30.0
fallback-moe path: gate_exps=- up_exps=- down_exps=q5_1
```

Raw completion is not a valid Gemma coherence gate right now. It can emit
repetition such as `is is is is`. The loop must use chat mode for this
effort.

## Reference implementations

The useful pattern is not a one-token microkernel retune. Production MoE
engines batch or group routed tokens by expert.

Read these local references before implementing grouped work:

- `/Users/zolotukhin/Workplace/llama.cpp/src/llama-graph.cpp`
  - `build_moe_ffn`: creates `selected_experts [n_expert_used, n_tokens]`,
    gathers router weights, calls `mul_mat_id` for gate/up/down, then
    weights and sums expert outputs.
- `/Users/zolotukhin/Workplace/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp`
  - `ggml_metal_op_mul_mat_id`: switches to grouped matrix-matrix when
    `has_simdgroup_mm`, row count is large enough, and `n_tokens >= 32`.
- `/Users/zolotukhin/Workplace/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`
  - `kernel_mul_mm_id_map0`: builds per-expert token counts and packed ids.
  - `kernel_mul_mm_id`: consumes that map with one grouped expert kernel.
- `/Users/zolotukhin/Workplace/vllm/vllm/model_executor/layers/fused_moe/`
  - `fused_moe.py`, `moe_align_block_size.py`, and router modules show the
    same topk -> sort/pack -> grouped expert matmul -> unpermute pattern.

## Already landed foundations

These are available on `main` and should not be reimplemented:

- `e235f69 Add Metal batched MoE top-k routing`
  - `src/shaders/metal/softmax_topk_batched.metal`
  - Writes `[token][expert_ids, normalized_weights]`.
- `faeb24f Add Metal MoE route packing primitive`
  - `src/shaders/metal/moe_route_pack.metal`
  - Builds per-expert counts and packed ids `token_idx * k + topk_slot`.

## Definition of "decent speed"

This effort should not claim success for a cosmetic improvement.

Minimum acceptable milestone:

- Gemma chat remains coherent on `What is the capital of France?`.
- `zig build test` passes.
- Official ReleaseFast verifier is coherent and reproducible.
- Decode stays above the old Debug plateau of `15 tok/s`.
- 20-token chat prefill stays materially below the original 90 second range.

Target milestone:

- Official ReleaseFast verifier reaches at least `50 tok/s`.
- Prefill no longer spends most wall time in per-token fallback MoE.
- Profile shows named remaining bottlenecks rather than generic
  `fallback-moe`.

Stretch milestone:

- Gemma 4 12B chat feels interactive for short prompts: first answer token
  in under 5 seconds for the France prompt and decode above `70 tok/s`.

## Execution order

### Step 0 - Keep the harness honest

Do not start by changing kernels. First make sure the loop is running the
right benchmark:

- `ZINC_MODEL_ID=gemma4-12b-q4k-m`
- `ZINC_PROMPT_MODE=chat`
- `ZINC_TEST_PROMPT="What is the capital of France?"`
- `ZINC_MAX_TOKENS=12`
- `ZINC_BUILD_OPTIMIZE=ReleaseFast`
- `ZINC_BENCHMARK_RUNS=3` for plateau work

The loop should classify `The capital of France is **Paris**.` as correct.
If it does not, fix the harness before optimizing.

Current Step 0 gate after cycle 38:

1. Confirm the build line says
   `Building (zig build -Doptimize=ReleaseFast)`.
2. Compare the loop verifier output with an immediate `--profile` run from the
   same binary.
3. If verifier and profile differ by more than about 20 percent, fix the
   measurement path before touching kernels.
4. If both report roughly `25-40 tok/s`, reset the mental baseline to that
   ReleaseFast number and ignore the old `15.89 tok/s` Debug plateau as a
   performance target.
5. If both report roughly `15 tok/s`, the agent's cycle-38 `36.55 tok/s`
   profile was not reproducible and must not be used as evidence.

### Step 1 - Establish a Gemma MoE parity switch

Before replacing the fallback path, add a validation mode that compares the
candidate Gemma MoE output against the existing fallback for one layer and one
token.

Required behavior:

- Gate with an env var, e.g. `ZINC_GEMMA_MOE_VALIDATE=1`.
- Run both paths for layer 0 at position 0.
- Compare the post-MoE vector before residual writeback.
- Report max abs diff and fail validation if it exceeds `1e-3`.

Do not trust text-only coherence for MoE routing. A one-expert misroute can
still produce plausible English for the short prompt.

### Step 2 - Add Gemma to the GPU-routed single-token MoE path

Current state:

- `canUseGpuRoutedBatchedMoe()` rejects `.gemma`.
- It also rejects `lt.ffn_gate_up_exps != null`.
- Gemma uses fused `ffn_gate_up_exps.weight`, GeGLU, shared expert, per-expert
  scaling, and Q5_1 expert down weights.

Implement a Gemma-specific GPU-routed decode path:

1. Reuse GPU router logits + `softmax_topk.metal`; avoid CPU topk readback.
2. Extend the routed path to understand `resolveMoeGateUpLayout()`, so the
   fused gate/up tensor can be addressed with gate and up offsets.
3. Use GeGLU for Gemma, not SwiGLU.
4. Add Q5_1 down support to the routed path using `dmmv_q5_1_moe.metal`.
5. Fold `ffn_down_exps_scale` into the expert weights exactly like the current
   fallback does.
6. Preserve shared expert gating and Gemma post-FFN norms.

Acceptance:

- Validation mode max abs diff below `1e-3` against the fallback.
- Chat output still contains Paris.
- Profile reduces CPU router/readback and/or commit count materially.

### Step 3 - Remove the one-commit-per-MoE-layer shape

The current profile shows thousands of command commits for a short answer.
After Step 2, the router and expert work should stay in the layer command
buffer rather than forcing a separate fallback command buffer per layer.

Allowed changes:

- Keep router, topk, expert gate/up/down, activation, weighted accumulate,
  post-norm, and residual add in one command buffer when data dependencies
  allow it.
- Use barriers only for real buffer dependencies.
- Keep CPU fallback available behind a guard until validation is stable.

Acceptance:

- `cmds` and `commits` in `--profile` drop materially for the same prompt.
- No correctness regression.

### Step 4 - Wire batched routing for Gemma prefill

Use the foundations already landed:

- `softmax_topk_batched.metal` for `[n_tokens, n_experts]` router logits.
- `moe_route_pack.metal` for per-expert counts and ids.

Add the host scratch:

- routing rows: `[n_tokens][2 * n_experts_used]` as u32
- expert counts: `[n_experts]` u32
- packed ids: `[n_experts][n_tokens]` u32
- per-expert intermediate and down-output buffers sized for `n_tokens * k`

Acceptance:

- A debug test routes a small two-token prompt and prints nonzero counts.
- Counts sum to `n_tokens * n_experts_used`.
- No runtime path uses this for production until Step 5 kernels exist.

### Step 5 - Implement grouped Gemma expert kernels

Do not jump directly to full simdgroup-matrix `mul_mat_id`. First land a
simple column-grouped DMMV that reuses one dequantized expert row for several
routed tokens.

Recommended first kernels:

- `dmmv_q4k_moe_cols.metal` or a Gemma-specific variant for fused gate/up.
- `dmmv_q5_1_moe_cols.metal` for expert down.

Shape:

- One expert id per `grid.y`.
- One packed-token block per `grid.z`.
- One output row block per `grid.x`.
- `NUM_COLS=4` first. Measure before trying 8.
- For each row, dequantize weights once, dot against up to 4 token vectors,
  and write `[expert_route_slot, row]`.

Why this shape:

- It is simpler than llama.cpp's full `kernel_mul_mm_id`.
- It still attacks the actual bottleneck: repeated expert weight reads across
  prompt tokens.
- It is a safe stepping stone to simdgroup-matrix kernels later.

Acceptance:

- Unit tests compare the grouped kernel against per-token DMMV for at least
  two tokens routed to the same expert.
- Max abs diff below `1e-3` for f32 outputs.

### Step 6 - Scatter weighted expert outputs back to token order

Add a scatter/accumulate kernel:

- Input: packed ids, counts, per-expert down outputs, routing weights.
- Output: `[n_tokens, hidden_dim]` MoE contribution.
- For each packed id, decode `token = id / k` and `slot = id % k`.
- Accumulate `routing_weight[token, slot] * expert_down`.

Acceptance:

- CPU reference test with two tokens and overlapping experts passes.
- Counts and ids are bounds-checked in debug mode.

### Step 7 - Enable Gemma batched prefill behind validation

Only after Steps 4-6:

- Relax Metal `canUseBatchedPrefill` for Gemma MoE behind an env flag first,
  e.g. `ZINC_GEMMA_BATCHED_PREFILL=1`.
- Reuse the dense Gemma batched prefill lessons from Effort 9:
  - embedding pre-scale
  - post-attention and post-FFN norms
  - per-layer head dimensions
  - proportional RoPE / SWA handling
  - `use_k_as_v` handling
- Add `ZINC_BATCHED_PREFILL=validate` coverage before enabling by default.

Acceptance:

- Last-token logits vs per-token path have max abs diff near float noise.
- Start with short chat prompts, then test longer prompts.
- Prefill time for the 20-token Gemma chat prompt drops significantly from
  the 90 second baseline.

### Step 8 - Only then consider simdgroup-matrix grouped matmul

If the column-grouped DMMV path is correct but still too slow, port the
llama.cpp shape more directly:

- token-per-expert counts
- packed route ids
- per-expert grouped matrix-matrix using `simdgroup_matrix`

This is higher risk and should not be the first implementation step.

## Post-cycle-38 execution path

The original Step 1 to Step 8 foundations have mostly landed. The remaining
work is no longer "make Gemma coherent"; it is "make the measured ReleaseFast
path coherent and faster." Use this order now:

1. Measurement reconciliation. Do not make model-kernel edits until the official
   verifier and `--profile` command use the same optimize mode and agree within
   normal noise.
2. Current bottleneck classification. Every agent cycle must quote the current
   `--profile` lines for `commitAndWait`, `record breakdown`, `dmmv bytes`, and
   `path bytes` before choosing an optimization.
3. If `commitAndWait` dominates and `cmds=commits=28`, reduce the remaining GPU
   work inside the 28 request commands. Do not chase command count unless the
   profile proves extra commands came back.
4. If `final` dominates record time, inspect the final norm plus LM-head path.
   Prior CPU Q8 scheduling, row-pairing, skip-store, and GPU argmax attempts
   already regressed, so only try a new LM-head change with before/after phase
   timing.
5. If `gpu-moe` or `moe-expert` bytes dominate, optimize the exact Q4_K/Q5_1
   MoE phase shown by profile. Avoid broad fused-expert rewrites unless there
   is a small isolated correctness test and a direct before/after profile.
6. If prefill remains near `9-10s` for the 20-token chat prompt after decode is
   stable, return to grouped batched prefill. Do not mix prefill and decode
   changes in one cycle.

Acceptance for future keeps:

- Correct Paris answer in chat mode.
- `zig build test` passes.
- Official verifier, not agent-only profile, improves by at least `1 tok/s`
  while stalled at this plateau, or a profile phase drops by at least 10 percent
  with no verifier regression.
- The self-analysis must name the phase it targeted and include before/after
  numbers from the same optimize mode.

## Known dead ends - do not repeat

These were already measured locally and should not be retried unless the
surrounding path has changed substantially:

- Q5_1 MoE down threadgroup width retunes: 128/256 thread variants regressed.
- Store-only MoE accumulate to remove zero fill regressed.
- No-logits zero-token prefill did not improve total time.
- GPU Q8 LM head for Gemma worsened total time because GPU wait rose.
- GPU post-MoE post-norm/residual tail regressed when tried as an isolated
  micro-change.
- Fused Q5_1 expert-down + weighted accumulate regressed.
- Broad Gemma GPU-routed decode MoE broke correctness in cycle 3.
- Grouped Q4_K column input-addressing changes regressed cycle 15.
- Gemma K-as-V V-unit-norm handling in batched prefill regressed cycle 18.
- Q5_1 four-rows-per-workgroup variants regressed cycles 24 and 28.
- CPU Q8 LM-head scheduling, row pairing, unused-logit skip-store, and GPU
  argmax variants regressed or broke tests in cycles 25, 26, 29, and 30.
- vLLM-style projection-to-activation fused expert kernel regressed badly in
  cycle 34.
- Shared expert dual Q4_K gate/up fusion regressed in cycle 35.
- 64-thread Q4_K fused gate/up groups regressed in cycle 37.
- K-cap/cached Q4_K widening beyond the current `K <= 3072` should not be tried
  unless profile identifies the `K=2816` large-M Q4_K path as the remaining
  bottleneck and excludes the known-bad `K=4096` case.

## Measurement gates

Every kept change must include:

```bash
zig build test
zig build -Doptimize=ReleaseFast
./zig-out/bin/zinc --model-id gemma4-12b-q4k-m \
  --prompt "What is the capital of France?" --chat -n 12 --profile
```

If a change touches generic Metal kernels, also run at least one Qwen smoke:

```bash
./zig-out/bin/zinc --model-id qwen3-8b-q4k-m \
  --prompt "What is the capital of France?" --chat -n 8
```

Reject a change if:

- It loses Paris coherence.
- It reduces correctness validation but only because a check was weakened.
- It moves work from profile-visible GPU time into unprofiled CPU work.
- It improves raw Gemma completions but regresses chat mode.
- It is justified only by an agent-side `--profile` speedup while the official
  verifier does not reproduce the gain under the same optimize mode.
- It lands within one-sample noise after cycle 38 without a measured phase
  reduction.

## Files likely to change

- `loops/implement_metal.ts` - harness config only; do not keep changing it
  after Step 0.
- `src/compute/forward_metal.zig` - Gemma MoE routing, scratch buffers, prefill
  orchestration.
- `src/shaders/metal/softmax_topk_batched.metal` - only if routing semantics
  are wrong.
- `src/shaders/metal/moe_route_pack.metal` - only if packed id layout needs
  extension.
- New grouped expert kernels under `src/shaders/metal/`.

## Expected end state

The final architecture should look like this:

```text
Gemma chat prompt
  -> batched hidden states [N,H]
  -> router logits [N,E]
  -> batched topk [N,K]
  -> route pack: counts[E], ids[E,N]
  -> grouped gate/up by expert and token block
  -> batched GeGLU
  -> grouped down by expert and token block
  -> scatter weighted outputs back to [N,H]
  -> post norms/residuals on GPU
```

For decode, the same semantics apply with `N=1`, but the win comes from
keeping the router/topk/expert/postnorm sequence GPU-routed and command-buffer
coherent rather than falling back through CPU readback and per-layer commits.
