# Qwen Models Support Breakdown

Last updated: 2026-03-29

This is an internal reference note for the next article. It is not polished blog copy. The goal is to capture, in one place, what made multi-model Qwen support difficult in ZINC, what turned out to be real versus misleading, and which fixes actually moved the system to a working state.

## Scope

Models covered in this pass:

- `Qwen3.5-2B-Q4_K_M.gguf`
- `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`

Primary runtime surfaces covered:

- CLI inference
- server inference and streaming chat
- deployment to the RDNA4 node
- managed model catalog and public model identity

Primary validation prompt:

- `The capital of France is`

Primary remote node:

- ``$ZINC_USER@$ZINC_HOST:$ZINC_PORT``

Known-good 35B baseline commit used as a reference:

- `8cac1ff6ff7e89c896e8207c54b63f5f4d41843c`

## Final Status As Of 2026-03-29

The short version is that both Qwen3.5 models are now working on the key regression checks, but it took more than one class of fix:

- model-family detection and metadata normalization
- tokenizer/BOS behavior
- packed Qwen attention layout
- SSM stability work and better diagnostics
- a subtle `Q5_K` dequant ordering regression that hit Qwen expert down projections
- server-side allocator ownership fixes that only appeared when the smaller model was actually deployed
- public UI/model-link fixes once the smaller model was live

Current validated state:

- 2B smoke: first token `11751`, output contains `Paris`
- 35B smoke: first token `11751`, output contains `Paris`
- live `9090` endpoint serves `Qwen3.5-2B`
- live streaming chat no longer fails with `Error: Failed to fetch`

## Why "multi-model Qwen support" was hard

Supporting another Qwen model did not mean swapping one GGUF path for another. The smaller and larger Qwen3.5 variants exercised different weak points in the runtime:

- The 2B model made the dense `qwen35` path matter. That path exposed tokenizer/BOS assumptions, SSM stability issues, and a server allocator bug that the CLI path did not reveal.
- The 35B-A3B model remained the best regression guard for MoE correctness, especially expert down projections and the general question of whether the runtime still matched a previously working baseline.
- Both models shared enough code that a fix for one could regress the other, but they diverged enough architecturally that "works on one Qwen3.5 model" was never a meaningful success condition.

In practice, "Qwen support" broke into three overlapping problems:

1. Model interpretation: understanding what the GGUF metadata actually meant.
2. Numerical/runtime correctness: getting the forward pass to produce the right tokens.
3. Operational correctness: making the server, catalog, UI, and deployment flow reflect the model that was actually loaded.

## The Architecture Problem Came First

The first trap was assuming that the Qwen3.5 family was simpler and more uniform than it was.

Important facts that mattered:

- `qwen35` is a dense hybrid model with SSM plus periodic full-attention layers.
- `qwen35moe` is a larger hybrid model with MoE behavior plus the same broader Qwen3.5 family quirks.
- In the codebase, architecture naming did not map cleanly to the model family names:
  - `src/model/loader.zig` parses `qwen35` into `Architecture.qwen35`
  - `src/model/loader.zig` still folds `qwen35moe` into `Architecture.qwen2_moe`

That historical naming was workable, but it made debugging more confusing than it should have been. The code organization lagged the actual model names, so "which path is this model really taking?" was not always obvious from the enum alone.

Two metadata fixes were especially important:

- `head_dim` had to come from `attention.key_length`, not `hidden_dim / n_heads`
  - Qwen3.5 uses `256` here, not the naive `2048 / 16 = 128`
- `full_attention_interval` had to come from GGUF metadata
  - the hybrid models do not run full attention on every layer

Without those fixes, RoPE sizing, KV sizing, projection dimensions, and the layer schedule all drifted away from the real model.

Relevant code surface:

- `src/model/loader.zig`
- `src/model/architecture.zig`
- `src/compute/forward.zig`

## Dense 2B And 35B-A3B Exercised Different Paths

This mattered more than expected.

The dense 2B model forced the `qwen35` path to work cleanly. In ZINC that meant:

- the dense hybrid graph builder path in `buildMambaDecodeGraph`
- dense Q plus separate gate support
- stable SSM behavior on the dense hybrid path
- correct tokenizer defaults for the Qwen3.5 family

The 35B-A3B model forced the MoE and mixed quantization path to stay honest. In practice it remained the best canary for:

- router correctness
- expert projection correctness
- especially `Q5_K` expert down-projection correctness

That split was useful later: if 2B was broken and 35B was fine, the bug was unlikely to be in the same place as if both were broken in the same way.

## Tokenizer And Prompt Construction Were More Important Than They Looked

One of the easiest ways to produce "kind of alive, kind of wrong" output is to get prompt construction slightly wrong.

### BOS behavior

The Qwen3.5 GGUFs used here did not always provide BOS metadata the way the older working path expected. In practice the models still wanted historical BOS behavior:

- BOS fallback had to be restored for the Qwen3.5 family
- `bos_id = 1` had to be assumed for `qwen35` and `qwen35moe` when the GGUF omitted BOS metadata
- `prepend_bos` had to default to true for the family

That logic now lives in `src/model/tokenizer.zig`.

### One allocator bug only appeared in server mode

The most instructive server-side Qwen bug had nothing to do with model math. It was an ownership bug.

`Tokenizer.encode()` allocates with `tokenizer.allocator`. In server mode, the route handlers were freeing the returned token slice with the per-request page allocator instead of the tokenizer allocator. That worked just well enough to be dangerous until the deployed 2B server hit a real request and crashed with:

```text
thread ... panic: incorrect alignment
...
/tmp/zinc-qwen35-clean/src/server/routes.zig:189:25
defer allocator.free(raw_tokens);
```

That produced the browser symptom:

- `Error: Failed to fetch`

The fix was structural, not cosmetic:

- prompt-token construction was centralized into `Tokenizer.encodePrompt()`
- `encodePrompt()` takes an output allocator and internally frees tokenizer-owned scratch correctly
- both CLI and server now use the same prompt-token path

That fix mattered because it removed an entire class of allocator mismatch, not just this one crash.

Relevant code surface:

- `src/model/tokenizer.zig`
- `src/server/routes.zig`
- `src/main.zig`

## Attention Layout Was A Real Qwen3.5-Specific Trap

The Qwen attention path had two different issues:

1. interpreting how Q and the attention gate are packed
2. deciding where the gate is actually applied

The important structural detail was this:

- Qwen3Next packs per-head blocks as `[Q(head_dim), gate(head_dim)]`
- this is per-head contiguous packing
- it is not element-interleaved packing

The decode loop had to:

- project into a temporary packed buffer
- split per-head Q and gate blocks out with explicit copy barriers
- keep the barrier ordering correct around those copies

There was also a semantic issue around where to apply the gate. The attention path was moved toward the reference behavior:

- `q_norm` and `k_norm` before RoPE
- gate handling aligned with the Qwen3Next reference path
- packed and separate-gate cases both supported

This class of issue was nasty because it could produce outputs that looked plausible enough to waste time. It did not necessarily crash anything.

Relevant runtime signals:

- packed Q/gate logging in `decodeStep`
- attention self-tests
- attention reference tests

Relevant code surface:

- `src/compute/forward.zig`
- `src/regression_tests.zig`

## The 2B SSM Blow-Up Was Real, But It Was Not The Final Root Cause

At one point the 2B model still generated garbage because the SSM recurrence itself was blowing up.

The diagnostic signature looked like this:

- layer-0 `delta_out` started small
- later in the same run it could climb from roughly `0.004` to `> 20,000`

That was a real bug. It forced us to get much more serious about SSM instrumentation:

- `SSM_DBG`
- `SSM_A_STATS`
- `ssm_norm.weight` shape/type logging
- gate, decay, and beta logging

Important observations from that stage:

- `ssm_a` was present and `f32`
- `ssm_norm.weight` was shared over the state dimension
- full attention and several DMMV paths were already matching references closely enough

That last point was useful. It meant "the model still talks nonsense" was not enough evidence to keep blaming flash attention or every projection kernel.

The dense `qwen35` path now stays on the CPU SSM reference path rather than the GPU SSM path:

- `src/compute/forward.zig` explicitly disables the GPU SSM path for `config.architecture == .qwen35`

This was a pragmatic stability choice. The dense 2B model needed a trustworthy path more than it needed a heroic GPU SSM path immediately.

Important nuance:

- The SSM instability was real.
- Fixing it was necessary.
- It was not sufficient to restore both models to the expected first-token behavior.

That distinction matters because it would have been easy to stop the investigation too early once `delta_out` stopped exploding.

## What Turned Out Not To Be The Main Blocker

Several things looked suspicious and consumed debugging time, but were eventually ruled out as the final blocker:

- flash attention math
- attention output projection
- several DMMV paths
- the general idea that "small Qwen is broken because the SSM math is still wrong"

What was verified as already correct enough:

- `ATTN_SELFTEST` at `seq_len=1`
- `ATTN_REFTEST` at `seq_len=5`
- DMMV checks for:
  - `wqkv`
  - `ffn_gate`
  - `ffn_up`
  - `attn_q`
  - `attn_output`
  - `ffn_down`

This was important because it narrowed the search surface. Once those checks were green, we needed a more model-specific explanation for why both models still missed the expected first token.

## The Real Final Regression Was Q5_K Ordering

This was the highest-value fix in the whole multi-model pass.

### Symptom

The best quick signal became:

- prompt: `The capital of France is`
- known-good first token: `11751`
- semantic meaning: `Paris`

On broken trees we saw several wrong first-token states over time, including:

- `524`
- `112`
- `264`
- `228`

Those numbers varied by exact tree state, but the invariant was simple:

- if `decode[0] != 11751`, the model was not actually back

### Why the 35B model was critical

The 35B model had previously worked on remote main at commit:

- `8cac1ff6ff7e89c896e8207c54b63f5f4d41843c`

That gave us a historical anchor. We were not debugging against a vague memory of "the output used to look better." We had a concrete first-token target and a known-good commit.

### Root cause

The regression was in `Q5_K` dequant ordering.

The bad version effectively treated the 64-element subgroup as interleaved:

- low nibble at `2 * e`
- high nibble at `2 * e + 1`

But GGML `Q5_K` dequantization expects contiguous halves inside each 64-element group:

- low nibble contributes to `y[l]`
- high nibble contributes to `y[32 + l]`

That detail mattered a lot for Qwen3.5 expert down projections. The MoE path in particular is very sensitive to getting those expert outputs numerically right.

The fix was applied in three places:

- CPU reference dequant in `src/compute/forward.zig`
- `src/shaders/dmmv_q5k.comp`
- `src/shaders/dmmv_q5k_moe.comp`

The source comments now explicitly say not to reintroduce the interleaved layout.

### Why this was easy to miss

This was not the sort of bug that always crashes or zeroes everything out.

It did something worse:

- it kept the model alive
- it let many internal checks still look sane
- it specifically poisoned a downstream path that strongly affected token choice

That is exactly the kind of bug that creates long, expensive false leads.

### After the fix

After restoring contiguous-half `Q5_K` ordering:

- 35B returned to the known-good first-token behavior from `8cac1ff`
- 2B also returned to first token `11751`
- both smoke tests again contained `Paris`

This was the point where "multi-model Qwen support" became real rather than partial.

## Reproducibility Was Part Of The Debugging Problem

One of the hardest non-code problems was baseline trust.

At one point `/root/zinc` on the node was not a clean baseline:

- it was on commit `2bb74e5`
- it also had a dirty local patch stack in inference files

That made it a terrible reference point.

This mattered because it is very easy to waste a day comparing:

- local `HEAD`
- a dirty remote worktree
- "remote main as I remember it"

The fix was procedural:

- treat `/root/zinc` as untrusted when dirty
- use a clean temp tree on the node such as `/tmp/zinc-qwen35-clean`
- keep referring back to the exact known-good 35B commit `8cac1ff...`

That may sound administrative, but it changed the debugging loop. Once the remote state was controlled, it became much easier to tell whether a fix actually restored behavior or merely changed it.

## First-Token Smoke Tests Turned Out To Be Extremely High Leverage

The single best practical test for this family became:

- prompt: `The capital of France is`
- inspect `decode[0]`
- expect token `11751`
- expect output text to contain `Paris`

Why this worked so well:

- it was fast
- it was stable across both target models
- it gave a binary signal long before long-form quality evaluation
- it caught both math regressions and serving regressions

We codified that into an opt-in smoke suite:

- `tests/test_qwen_smoke.ts`
- `tests/test_qwen_smoke.test.ts`

The smoke suite now validates both:

- `Qwen3.5-2B-Q4_K_M.gguf`
- `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`

with the same prompt and expected first token.

## Source-Level Regression Guards Were Worth Adding

The bugs here were not all naturally covered by ordinary unit tests. Some were "the code should continue to contain this exact structural decision" style bugs.

That is why `src/regression_tests.zig` now carries source-level guards for:

- packed Q/gate split barriers
- gate placement on the attention path
- compute-to-transfer barrier before KV writes
- layer-boundary compute barrier after FFN residual
- `Q5_K` contiguous-half ordering in both shaders
- chat UI model-link behavior

This is not elegant in the abstract, but it is cheap insurance against exactly the kind of regression that already happened once.

## Public Serving Exposed Additional Multi-Model Problems

Once the smaller model was actually deployed publicly, two more problems appeared immediately.

### 1. The UI still linked to the 35B Hugging Face page

The chat page originally hardcoded:

- `https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF`

That meant the UI could show the 2B model name but still link to the wrong model card.

The fix:

- derive the link from the reported model name
- update the model tag from `/health` and `/v1/models`

This is a small bug, but it matters when a public deployment is explicitly being used to validate smaller-model support.

### 2. The deployed chat endpoint crashed on real requests

This was the allocator bug described earlier. It only became obvious when a real streaming request hit the deployed 2B server.

The sequence was:

- user sends a chat request
- browser reports `Error: Failed to fetch`
- remote log shows an `incorrect alignment` panic in the route handler

That was a good reminder that "the model loads and answers a smoke prompt" is not the same thing as "the serving path is production-safe."

## Managed Multi-Model Support Needed Its Own Plumbing

Beyond inference correctness, the repo now carries explicit managed-model metadata for both Qwen3.5 targets.

Catalog entries now exist for:

- `qwen35-2b-q4k-m`
- `qwen35-35b-a3b-q4k-xl`

The catalog records:

- display name
- family
- quantization
- download URL
- homepage URL
- SHA256
- size
- VRAM budget

This matters for two reasons:

1. It turns "supports Qwen" into something operationally concrete.
2. It forces the project to acknowledge that smaller and larger Qwen variants have different footprint and deployment assumptions.

Relevant code surface:

- `src/model/catalog.zig`
- `src/model/managed.zig`

## Useful Evidence And Hard Signals

These were the most useful hard signals during the debugging cycle.

### Signals that something was broken

- `decode[0]` not equal to `11751` on `The capital of France is`
- 2B or 35B producing wrong early token IDs like `112`, `228`, `264`, `524`
- 2B `SSM_DBG` `delta_out` L2 growing from tiny values to very large values
- live chat returning `Error: Failed to fetch`
- remote log showing `incorrect alignment`

### Signals that a suspected subsystem was actually okay

- attention self-test matched
- attention reference test matched
- DMMV checks matched CPU reference closely enough
- `delta_out` stayed bounded again after the SSM stabilization work

### Signals that the final Q5_K fix had actually landed

- 35B back to the historical first-token behavior from `8cac1ff`
- 2B also producing first token `11751`
- smoke outputs containing `Paris`

## What Was Solved Versus What Was Just Bounded

### Solved

- Qwen3.5 family detection for the target models in this repo
- correct `head_dim` extraction for Qwen3.5
- BOS fallback behavior for the family
- packed Q/gate handling for the attention path
- `Q5_K` ordering regression in CPU reference and shaders
- server allocator crash
- public UI wrong-model link
- managed model entries for both 2B and 35B
- first-token smoke coverage for both target models

### Bounded or reduced, but not "done forever"

- dense `qwen35` SSM path confidence
  - stable enough for current smoke and deployment
  - still benefits from broader prompt evaluation
- code organization clarity
  - enum naming and historical path names are still a little misleading
- server/UI model identity
  - current link mapping is string-based, not yet driven by a single canonical model catalog surface

## What We Learned

These are the higher-level takeaways worth reusing in the article.

### 1. Smaller-model support is not "easy mode"

The instinct is that a 2B model should be simpler than a 35B-A3B MoE model. In practice the smaller model exercised different paths:

- dense hybrid path
- tokenizer assumptions
- serving path
- SSM stability choices

It was not a reduced version of the same problem.

### 2. The best regression test was embarrassingly small

The most useful cross-model test was one prompt and one first token:

- `The capital of France is`
- expect `11751`

That was more valuable than a lot of vague qualitative evaluation.

### 3. Internal self-consistency checks are necessary but not sufficient

Attention and DMMV checks passing did not mean the model was back. They only meant those components were no longer the best suspect.

### 4. Operational bugs matter once you actually support more than one model

If the server crashes on live chat requests or the UI still points to the wrong model card, then "multi-model support" is still incomplete even if the forward pass is mathematically correct.

### 5. Code comments and source-level guards are justified here

The `Q5_K` regression was too easy to reintroduce. It needed:

- code comments in the dequant paths
- direct regression tests guarding against the bad `2u * e` pattern

## Useful Commands

Remote CLI smoke:

```bash
ssh -p "$ZINC_PORT" "$ZINC_USER@$ZINC_HOST" \
  'cd /tmp/zinc-qwen35-clean && \
   RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
   --model /root/models/Qwen3.5-2B-Q4_K_M.gguf \
   --prompt "The capital of France is" \
   --max-tokens 8'
```

```bash
ssh -p "$ZINC_PORT" "$ZINC_USER@$ZINC_HOST" \
  'cd /tmp/zinc-qwen35-clean && \
   RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
   --model /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
   --prompt "The capital of France is" \
   --max-tokens 8'
```

Bun smoke suite:

```bash
ZINC_QWEN35_2B_MODEL=/root/models/Qwen3.5-2B-Q4_K_M.gguf \
ZINC_QWEN35_35B_MODEL=/root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
bun test tests/test_qwen_smoke.test.ts
```

Live server health:

```bash
curl -fsS "http://$ZINC_HOST:9090/health"
```

## Best Candidate Narrative For The Article

If this turns into a full article, the cleanest structure is probably:

1. "Supporting another Qwen model was not a model-download problem. It was a systems problem."
2. Show how the smaller and larger Qwen3.5 variants stressed different parts of the engine.
3. Walk through the false leads:
   - attention looked suspicious but was not the final blocker
   - SSM blow-up was real but not the final blocker
4. Land on the real regression:
   - `Q5_K` ordering in expert down projections
5. End on the broader lesson:
   - multi-model support only becomes real when runtime math, tests, server behavior, and public model identity all agree

## One-Sentence Summary

Multi-model Qwen support in ZINC was hard because the 2B and 35B variants did not just differ in size; they exposed different assumptions in architecture parsing, prompt construction, hybrid attention/SSM execution, quantized expert projections, and even the serving layer, and the system only became trustworthy once all of those surfaces were validated together.
