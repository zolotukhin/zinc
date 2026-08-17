# Effort 15 - RDNA4 Qwen 3.6 27B dense-hybrid prefill and decode

> **Status:** server-vs-server harness fixed · Qwen3.6 27B now beating reference on the RDNA matrix · post-reboot `decode-extended` prefill 324.37 vs 253.17 tok/s

Date: 2026-05-18

Target model:

- RDNA node GGUF: `/root/models/Qwen3.6-27B-Q4_K_M.gguf`
- Harness id: `qwen36-27b-q4k-m`
- Architecture shape: dense Qwen 3.6 hybrid with SSM layers, not MoE.
- Important tensor/log facts seen on the RDNA path:
  - `hidden_dim = 5120`
  - `intermediate_dim = 17408`
  - `output.weight = q6_k`, dims `(5120, 248320)`
  - SSM fastpath line: `ssm qkv=q6_k gate=q4_k alpha=f32 beta=f32 conv=f32 out=q5_k`
  - Dense FFN: `ffn_gate=q4_k`, `ffn_up=q4_k`, `ffn_down=q6_k`

## Executive summary

Latest checkpoint: the RDNA performance suite now measures ZINC through
one reusable server per model, matching the reference server residency
model. Under that server-vs-server contract, clean ZINC beats the
reference on Qwen 3.6 27B overall and on the target `decode-extended`
prefill metric. The earlier `~146 tok/s vs 252 tok/s` result was a
mismatched harness comparison: one-shot ZINC CLI samples against a warmed
persistent reference server.

The dominant kernel budget is still dense FFN, then SSM, and the
single-shader micro-fusions tested on this exact model remain mostly
mixed or negative. Do not treat the harness fix as proof that dense-down
is fully optimized; treat it as the corrected measurement baseline.

Latest patched-suite artifact:

- `/tmp/zinc-rdna1-full-suite-20260629-082140.json`
- ZINC provenance: `f719ccccb501`
- Baseline provenance: `llama-server` `9725a313b`

Post-reboot Qwen 3.6 27B:

| scenario | ZINC prefill | reference prefill | ZINC decode | reference decode |
|---|---:|---:|---:|---:|
| core | 212.73 | 184.18 | 31.98 | 30.65 |
| context-medium | 388.21 | 73.03 | 31.72 | 30.47 |
| context-long | 408.64 | 74.83 | 31.71 | 30.50 |
| decode-extended | 324.37 | 253.17 | 31.64 | 30.40 |

Full RDNA1 post-reboot summary:

| model | ZINC prefill | reference prefill | ZINC decode | reference decode | overall |
|---|---:|---:|---:|---:|---:|
| Qwen 3.6 35B-A3B | 546.21 | 397.22 | 167.06 | 108.18 | 152.7% |
| Qwen 3.6 27B | 212.73 | 184.18 | 31.98 | 30.65 | 104.9% |
| Qwen 3.5 9B | 735.53 | 549.17 | 97.30 | 85.49 | 114.8% |

The two Gemma rows in this run are baseline-only because ZINC's server API
returned without emitting the `Generated` timing log that the fair harness
requires. The harness now records that as unavailable instead of waiting for
the full command timeout.

Previous patched-suite artifact:

| scenario | ZINC prefill | reference prefill | ZINC decode | reference decode |
|---|---:|---:|---:|---:|
| core | 216.63 | 184.01 | 32.02 | 30.64 |
| context-medium | 391.40 | 75.60 | 31.78 | 30.48 |
| context-long | 409.93 | 75.20 | 31.74 | 30.51 |
| decode-extended | 325.20 | 252.66 | 31.62 | 30.41 |

Historical target table from the pre-layer-major, CLI-measured baseline:

| scenario | ZINC prefill | llama prefill | prefill uplift needed | ZINC decode | llama decode | decode uplift needed |
|---|---:|---:|---:|---:|---:|---:|
| core | 17.06 | 61.12 | 3.58x | 20.79 | 34.43 | 1.66x |
| context-medium | 24.15 | 195.01 | 8.07x | 29.42 | 34.40 | 1.17x |
| context-long | 26.04 | 69.89 | 2.68x | 30.17 | 44.33 | 1.47x |
| decode-extended | 19.04 | 97.29 | 5.11x | 25.82 | 31.29 | 1.21x |

The realistic milestone order:

1. Establish a clean 27B-only baseline and profile artifact.
2. Add a validation-only layer-major prefill harness for this SSM+dense
   shape without replacing production output.
3. Batch dense FFN prefill per layer and token chunk, because it is the
   largest per-token weight stream and does not have the SSM recurrent
   dependency.
4. Batch SSM projections per layer and token chunk, but keep delta/state
   recurrence exact until a validated layer-major state path exists.
5. Only then chase decode improvements; keep those flag-gated and judged
   by the full four-scenario matrix.

## Current measured standing

Artifact: `/tmp/zinc-rdna-qwen36-27b-20260517-120205.json`

This run compared ZINC to llama.cpp on the same RDNA node using the
four public scenario shapes.

| scenario | ZINC prefill tok/s | ZINC decode tok/s | llama prefill tok/s | llama decode tok/s | ZINC prefill pct | ZINC decode pct |
|---|---:|---:|---:|---:|---:|---:|
| core | 17.06 | 20.79 | 61.12 | 34.43 | 27.9% | 60.4% |
| context-medium | 24.15 | 29.42 | 195.01 | 34.40 | 12.4% | 85.5% |
| context-long | 26.04 | 30.17 | 69.89 | 44.33 | 37.3% | 68.1% |
| decode-extended | 19.04 | 25.82 | 97.29 | 31.29 | 19.6% | 82.5% |

Artifact: `/tmp/zinc-rdna-qwen36-27b-after-20260517-121618.json`

Clean ZINC-only rerun:

| scenario | ZINC prefill tok/s | ZINC decode tok/s |
|---|---:|---:|
| core | 17.49 | 21.21 |
| context-medium | 23.97 | 29.43 |
| context-long | 26.99 | 31.80 |
| decode-extended | 19.14 | 26.95 |

Interpretation:

- Short `core` prefill is a 5-token prompt and has TTFT noise. Keep it
  in the matrix, but do not optimize from it alone.
- `context-medium`, `context-long`, and `decode-extended` show the real
  prefill problem: ZINC does not amortize prompt tokens.
- Decode is closer than prefill but still behind. The gap ranges from
  17% to 40% depending on scenario, with long-context decode the most
  painful.

2026-05-20 loop checkpoint:

- Best accepted effort-15 checkpoint on the current site-aligned Coding
  Review prefill benchmark is `49.91 tok/s` (`7cec485`), after moving the
  Qwen3.6-27B dense prefill segment range into the late layer-major path.
- The old synthetic Paris prompt can still report about `148 tok/s`, but
  it is not the controlling workload and should not drive keep/revert
  decisions.
- The current biggest profile bucket after the recent keeps remains
  `dense_ffn` (recent refreshes around `2.4 s` on the site-aligned
  prompt). Segment boundary additions near the lower layers are now
  repeatedly measured dead; use fresh subphase data before editing.

2026-05-21 loop checkpoint:

- Best accepted effort-15 checkpoint on the same site-aligned Coding
  Review prefill benchmark is now `64.87 tok/s` (`5610c871`), after:
  - default-on tiled Q6_K dense-down prefill (`7a423f3`, cycle 27)
  - default-on layer-major SSM batched-delta prefill (`4b9ff03`,
    cycle 29)
  - scoped layer-major dense-FFN barriers (`5610c871`, cycle 32)
- The current prefill profile is no longer a single obvious hotspot:
  `dense_ffn ~= 1848 ms`, `ssm ~= 1498 ms`, `attn ~= 433 ms`.
  Dense subphases are roughly tied (`gateup ~= 934 ms`,
  `down ~= 916 ms`), while the largest single SSM subphase is still
  projection (`ssm_proj ~= 1095 ms`). `ssm_delta` is no longer the main
  SSM cost after the batched-delta keep.
- Post-64.87 dead ends are informative: Q4/Q6 tile retunes, scoped
  barrier variants, dense-FFN submit fusion, Q6 dequant hoists, and
  tail-column FMA skipping did not produce the next jump. Treat
  sub-1% samples as noise unless a paired old-vs-new control and a
  phase/profile counter both point the same way.
- The next credible jump probably needs either a shaderstats-backed
  dense kernel change that explains VGPR/occupancy/memory behavior, or
  a validated SSM projection dataflow change. Another blind tile-shape
  sweep is unlikely to clear the new `+~0.65 tok/s` keep threshold.
- After cycle 85, the controller had `best=64.87` but current code had
  fallen to about `63.28 tok/s`. Root cause: rejected cycles were only
  reverting `src/`, so a `build.zig` shader-install edit could leak
  across cycles and make the local checkout diverge from the accepted
  checkpoint. The source/build tree was restored to the rebased cycle-32
  source (`dba5bdf`), and the controller should revert both `build.zig`
  and `src/` for future rejected agent changes.
- After the cycle-94 restart check, clean builds of the saved best
  commit did not install `mul_mm_q6k.spv` even though the accepted Q6_K
  dense-down path probes that pipeline at runtime. Some previous remote
  measurements depended on stale files in `zig-out/share/zinc/shaders`.
  The harness now clears the installed shader directory before each
  remote build, and `build.zig` installs `mul_mm_q6k` explicitly so the
  dense-down path is reproducible instead of artifact-dependent.

2026-05-24 50-cycle codex run post-mortem:

- A 50-cycle codex run (`--effort 15 --model qwen3627b`) was driven off a
  fresh post-shader-clean baseline of `58.83 tok/s` on the site-aligned
  Coding Review prefill prompt. It reported `best=79.63 tok/s`
  (`bestCycle=44`, HEAD `63dc951`, `+35.4%`). Run shape: 6 perf keeps,
  1 foundation keep, 37 reverts, 5 no-op cycles.
- Real accepted staircase: cyc4 `59.89`, cyc9 `68.06` (default-on
  layer-major SSM projection/conv), cyc39 `69.89`, cyc40 `72.06` (edits to
  the already-installed `mul_mm_q4k`/`mul_mm_q6k`), cyc43 `73.98`,
  cyc44 `79.63`.
- **The `79.63` headline is almost certainly a stale-shader artifact and
  must not be quoted as the reproducible number.** Cycles 43 and 44 added
  `src/shaders/mul_mm_q6k_full.comp` and
  `src/shaders/mul_mm_q4k_gate_up_swiglu_full.comp`, but neither is in the
  `shader_sources` tuple in `build.zig`, so glsl→spv never compiles or
  installs them. A clean build loads the older fallback kernels, meaning
  the cyc44 benchmark did not run the branchless full-tile path it claimed
  to add.
- Corroboration: cycle 47 did exactly the missing install ("so clean RDNA
  builds load the existing fast paths"), and the number dropped to
  `63.71 tok/s`, so it was reverted as a regression. The branchless
  full-tile path is therefore a regression versus the installed kernels,
  and the recorded staircase from cyc43 on overstates reproducible perf.
- This is the same failure class as the violent `~59 ↔ ~68 tok/s`
  oscillation in cycles 9–30, where cycles 12/16/21/25 each re-discovered
  that new batched SSM shaders were not installed. The `eb6c768` fix
  (clear install dir + install `mul_mm_q6k`) was incomplete: **every new
  `.comp` must be hand-added to `shader_sources` in `build.zig`, and the
  agent repeatedly forgot.** Until that is automated, every measured win
  that adds a new shader is suspect.
- `bandwidthUtil` was `0.0` for all 50 cycles — the metric is dead and
  gives the loop no roofline signal.
- Cycles 45–50 were unproductive: 45–47 regressed below `79.63`, and
  48–50 were no-ops because codex hit its usage limit (resets
  2026-05-26 11:33). Those 3 dead cycles still incremented `stall` and
  triggered the cycle-50 pivot.

## Decode phase budget

A clean single-stream profile on a 48-token decode prompt produced:

```
Generated 48 tokens at about 27.23 tok/s
avg GPU decode token = 35.56 ms

Top-level phases:
  attention   2.27 ms
  ssm        12.46 ms
  dense_ffn  18.29 ms
  tail        1.90 ms

SSM subphases:
  proj        5.39 ms
  conv        0.52 ms
  delta       4.38 ms
  gnorm       0.36 ms
  out         2.20 ms

Dense FFN subphases:
  gateup     10.96 ms
  down        7.35 ms
```

Approximate decode share:

| bucket | ms/token | share |
|---|---:|---:|
| dense FFN | 18.29 | 51% |
| SSM | 12.46 | 35% |
| attention | 2.27 | 6% |
| final tail | 1.90 | 5% |

This model is not primarily an attention problem. Attention work matters
for long context, but the first decode improvements should target dense
FFN and SSM. A flash-attention-only plan will not close the 27B gap.

## Why prefill is slow today

The relevant entry point is `prefillBatched` in `src/compute/forward.zig`.
It calls `canUseBatchedPrefillRdna` before using the batched RDNA
orchestration. That gate currently rejects:

```zig
if (cfg.n_experts > 0) return false;
if (cfg.ssm_d_inner > 0) return false;
```

For Qwen 3.6 27B, `cfg.ssm_d_inner > 0`, so the model falls back to
`prefillBatch`. That fallback processes prompt tokens through the same
token-major path used by decode. Every prompt token rereads the dense
FFN and SSM projection weights.

The current batched prefill implementation is a good fit for dense
attention-only/Gemma-like paths: it keeps `scratch_hidden` as
`[n_tokens, hidden_dim]` and runs layer-major batched attention and dense
FFN. It is not yet a safe fit for Qwen 3.6 27B because SSM layers carry
state and because every layer's SSM branch feeds the hidden stream before
the layer's dense FFN and before later layers.

The dependency that must be preserved for each layer and token is:

```text
hidden[token]
  -> attn_norm
  -> attention OR SSM branch
  -> hidden[token] += branch_output
  -> ffn_norm
  -> dense FFN
  -> hidden[token] += ffn_output
  -> next layer
```

For SSM layers, this is stricter:

```text
hidden[token]
  -> SSM projections (qkv/z/alpha/beta)
  -> conv/state recurrence in token order
  -> gated norm + ssm_out
  -> hidden[token] += ssm_output
  -> dense FFN for the same layer/token
```

Skipping the SSM output for each token while trying to run a single
post-loop batched SSM update corrupts `hidden[token]`, which then
corrupts later projections and the FFN. This exact problem is already
documented in `runSsmLayerGpu`: the prior production skip was removed
because it produced incoherent hidden evolution.

## Why decode is slow today

Dense FFN weight traffic per layer is large:

```text
gate q4_k: 17408 x 5120 weights x 144/256 bytes ~= 50 MB
up   q4_k: 17408 x 5120 weights x 144/256 bytes ~= 50 MB
down q6_k:  5120 x 17408 weights x 210/256 bytes ~= 73 MB

dense FFN stream per layer ~= 173 MB, before activation reads/writes
```

Every layer has the dense FFN. Even if the exact layer count changes by
checkpoint, the decode profile confirms this traffic is the largest
bucket: `dense_ffn=18.29 ms`, with `gateup=10.96 ms` and `down=7.35 ms`.

SSM also has meaningful weight traffic plus recurrent state work:

```text
wqkv: q6_k, M=conv_channels, K=hidden_dim
z:    q4_k, M=d_inner,       K=hidden_dim
alpha/beta: f32, small M=dt_rank
out:  q5_k, M=hidden_dim,    K=d_inner
delta_net: recurrent state update, not just matvec traffic
```

For decode, there is no cross-token batch to amortize weight reads:
autoregressive dependency forces one generated token at a time. That is
why decode wins must come from fewer dispatches, better single-token
kernel shape, or cutting unnecessary memory traffic. Re-reading weights
once per token is fundamental unless speculative/multi-token decoding is
introduced, which is outside this effort.

## Failed or rejected attempts on this exact 27B target

Do not repeat these without a new measurement hypothesis and a clear
reason the prior failure mode no longer applies.

| attempt | result | decision |
|---|---|---|
| Widen dense fused gate+up+SwiGLU eligibility to include `inter_dim=17408` | mixed; core/context-long worse, only some decode rows looked better | rejected |
| Dense-delta gate variant | mixed; medium/long decode worse | rejected |
| `ZINC_SSM_DELTA_COLS8=0` default flip | quick profile sometimes faster, full matrix mixed/worse | rejected |
| Q6_K+Q4_K fused SSM qkv+z pair | engaged but regressed; SSM proj got slower | rejected |
| Q6_K K=17408 specialization for dense FFN down | regressed decode and dense_ffn | rejected |
| Wider Q4/Q6 variants for dense FFN/down | flat or negative across matrix | rejected |
| Widened fused attention o-proj merge to `hidden_dim=5120` | severe long-context regression, attention exploded | rejected |
| `ZINC_SSM_DELTA_NORMED_QK=1` | regressed | rejected |
| Full SSM batched prefill gate flip | GPU hard resets / `QueueSubmitFailed`; dependency bug in hidden evolution | hard ban |
| Prefix SSM projection replay (`ZINC_QWEN36_27B_SSM_PREFILL_PROJ=z/qkv/both`) | coherent after keeping fused RMS+alpha+beta, but context-medium regressed from ~29.24 tok/s to 29.09-29.17 tok/s at L1 and 27.91-28.97 tok/s at L2/L4/L8 | default-off diagnostic only |
| Forcing separate SSM alpha/beta path (`ZINC_FUSED_SSM_AB=0`) | immediate `QueueSubmitFailed` on Qwen 3.6 27B | do not use as a prerequisite for 27B prefill work |
| Layer-major dense segment lower-bound additions | layer-1 extension measured old 4-62 override at `50.16 tok/s` vs new layer-1 schedule at `50.07 tok/s`; earlier prefix/layer-depth sweeps were flat or worse | rejected unless a new profile proves lower layers are hot |
| Partial hidden scratch copy fused with first attention RMS norm at segment handoff (`ZINC_QWEN36_27B_PARTIAL_ATTN_NORM_STORE`) | OFF median `49.96 tok/s` `[51.32, 49.82, 49.96]`; ON median `49.53 tok/s` `[49.53, 49.42, 49.62]`; also moved attention RMS outside the normal phase timer | rejected |
| Branchless full-tile Q6_K GEMM (`mul_mm_q6k_full.comp`) and Q4_K gate/up/SwiGLU (`mul_mm_q4k_gate_up_swiglu_full.comp`) for dense prefill (cyc43/cyc44) | reported `73.98`→`79.63 tok/s`, but the new shaders were never added to `shader_sources` in `build.zig`; clean builds ran fallback kernels. When the install was actually wired (cyc47) the number fell to `63.71 tok/s` | rejected as stale-shader artifact; delete the orphan `.comp` files or re-measure honestly with them installed |
| Dense gate/up Q8 output interleaved by K-block for exact 64-token dense-down | enabled with `ZINC_QWEN36_27B_INTERLEAVED_DOWN=1`; profiled `64 tokens in 472.1 ms (135.57 tok/s)`, with dense down around `224 ms` vs roughly `105 ms` default | rejected; worse activation layout for current dense-down kernels |
| Disabling the BM64 exact dense-down accumulate sibling | enabled with `ZINC_QWEN36_27B_BM64_DOWN_ACC=0`; profiled `64 tokens in 371.5 ms (172.29 tok/s)` during one run, but dense down split was worse (`q4=57.0 ms`, `q6=55.3 ms`) than default | rejected; do not use as a keep without full-suite evidence |
| Dense-down residual accumulation opt-out | enabled with `ZINC_QWEN36_27B_DENSE_DOWN_ACC=0`; clean profile showed `149.15 tok/s` and added `4.7 ms` residual_acc, vs default `175.43 tok/s` profile in the same build | rejected; fused residual write is faster despite read-modify-write |
| Forcing DP4a pipelines to request subgroup 32 | enabled with `ZINC_DP4A_WAVE32=1`; one stale/profile run looked promising, but clean rebuild profile fell to `152.60 tok/s` and quick steady samples stayed near `150-153 tok/s` | rejected for this metric unless a later shaderstats run explains and fixes the instability |
| Low-register MMQ64-style exact dense-down shader | enabled with `ZINC_QWEN36_27B_MMQ64_DOWN=1`; clean exact-prompt samples were `145.49, 141.22, 146.63 tok/s` vs current default steady `151.50, 151.42 tok/s`; adding `ZINC_DP4A_WAVE32=1` still settled around `148.06, 143.89 tok/s` | rejected; lower per-lane register footprint did not offset worse memory/reuse behavior |
| RDNA DPM pre-command lock as a speed lever | fixed the suite shell prelude, but the active RDNA card exposed only one memory-clock line and remained at `auto`; suite artifact `/private/tmp/zinc-rdna1-qwen36-27b-command-dpm-suite-r3w1-20260628221155.json` measured `decode-extended=141.85 tok/s` vs accepted `146.95 tok/s` | keep the harness hardening only; do not treat DPM locking as a performance win on this node |
| Prefix-depth retune for exact 64-token prompt | swept `ZINC_QWEN36_27B_DENSE_PREFILL_LAYERS={1,2,3,4,5,6,8,12}`; steady samples stayed near `149-153 tok/s` for 1-3 and got worse after 4 (`~146`, `~142`, `~137 tok/s`) | rejected; default prefix split remains best-enough for this metric |
| Disabling default-on layer-major / DP4a gates for exact 64-token prompt | default steady `~149-151 tok/s`; `ZINC_QWEN_DENSE_FFN_DP4A=0` fell to `~89 tok/s`, `ZINC_QWEN36_27B_DP4A_DOWN=0` to `~74 tok/s`, `ZINC_QWEN36_27B_SSM_BATCHED_DELTA=0` to `~69 tok/s`, `ZINC_QWEN36_27B_DENSE_PREFILL=0` to `~26 tok/s` | rejected; the accepted layer-major/DP4a structure is still mandatory |

Rejected matrix details:

| artifact | core | context-medium | context-long | decode-extended |
|---|---:|---:|---:|---:|
| clean ZINC | 17.49 / 21.21 | 23.97 / 29.43 | 26.99 / 31.80 | 19.14 / 26.95 |
| fused-wide rejected | 17.05 / 20.61 | 23.48 / 30.66 | 25.29 / 27.70 | 18.69 / 27.16 |
| dense-delta rejected | 18.21 / 21.10 | 23.98 / 27.66 | 25.81 / 30.87 | 19.12 / 27.00 |
| q6wide rejected | 16.32 / 20.10 | 23.58 / 28.94 | 25.72 / 28.03 | 18.27 / 26.76 |
| q4wide rejected | 17.41 / 20.85 | 24.04 / 28.80 | 25.88 / 31.88 | 18.68 / 25.86 |
| wide rejected | 16.49 / 22.70 | 23.53 / 30.05 | 25.74 / 28.89 | 18.16 / 26.40 |

Each cell is `prefill tok/s / decode tok/s`.

2026-05-18 follow-up:

- Default dense-FFN prefix depth remains one layer for context prompts.
  The real coding-review prompt measured ~29.23-29.24 tok/s prefill at
  L1; L2/L4/L8 with SSM projection replay were all slower.
- Very short prompts are the one safe exception. The 5-token raw smoke
  prompt improved from roughly 18-20 tok/s at L1 to ~23.9-24.1 tok/s at
  L8 with identical output tokens. Production now chooses L8 only when
  `prompt_len <= 8`; longer prompts keep L1.
- The SSM qkv/z replay path is left behind
  `ZINC_QWEN36_27B_SSM_PREFILL_PROJ={qkv,z,both}` for diagnostics. It is
  not a fix for the context prefill gap because the replay setup/copy
  overhead cancels the saved projection work.

2026-05-20 interrupted-cycle analysis:

- I stopped the loop after it had produced a candidate source edit, then
  measured the candidate directly against its own control instead of
  allowing more cycles to build on it.
- Candidate: fuse partial hidden scratch copy and first attention-layer
  RMS norm at a full-attention segment handoff.
- Result: control/OFF median `49.96 tok/s`; candidate/ON median
  `49.53 tok/s`. The candidate was reverted.
- Lesson: do not keep moving work out of the normal `attention`/`dense_ffn`
  profile sections for cosmetic copy removal. It can make phase budgets
  less trustworthy while not moving the real prefill wall time.
- Next loops should pivot from boundary-extension variants to
  measurement-backed dense subphase work: shaderstats for the current
  fused Q4_K gate/up/SwiGLU path and Q6_K batched down/kpar path, then
  one code change tied to the top subphase.

2026-06-28 live session notes:

- Cold CLI-vs-server comparison artifact for `decode-extended` prefill:
  baseline median `252.70 tok/s` vs ZINC CLI median `146.95 tok/s`.
  This is no longer the decision metric because the two engines did not
  have the same backend residency.
- Clean current quick samples after rebuilding on RDNA for the exact
  64-token prompt: `178.04, 149.87, 150.80, 153.04, 151.51 tok/s`.
  Treat the first sample as warm cache/setup outlier; steady state remains
  around `150-153 tok/s` for one-shot CLI.
- Reference persistent-server probe on the same prompt reported
  `cache_n=0` and log lines saying full prompt reprocessing was forced for
  this hybrid/recurrent model, so the reference prefill number is not a
  prompt-cache artifact.
- ZINC persistent-server probe on the same prompt reached stable post-warm
  prefill around `347-350 tok/s` with the current clean build. The patched
  full suite now measures ZINC through one reusable server per model; the
  `decode-extended` result is ZINC `325.20 tok/s` prefill vs reference
  `252.66 tok/s`, and decode `31.62` vs `30.41 tok/s`.
- Current clean profile for the same prompt: `64 tokens in 364.8 ms
  (175.43 tok/s)`, with GPU phase totals `dense_ffn=176.2 ms`,
  `ssm=121.0 ms`, `attn=22.6 ms`, `tail=2.2 ms`.
- Dense subphases remain split roughly `gateup=70.6 ms`, `down=105.5 ms`.
  SSM projection remains `proj=69.7 ms` (`qkv=56.6 ms`, `z=7.6 ms`,
  `out=24.3 ms`). The next target must remove substantial projection
  matmul time; residual-add and layout micro-switches are too small or
  negative.
- A K=5120 BM64/BN64 exact SSM projection specialization moved the profile
  only slightly (`~368.9 ms` to `~362-365 ms` depending on run) and did not
  materially move steady `decode-extended` prefill. Keep only if a full
  suite later proves it, otherwise treat it as diagnostic.
- Reference-code inspection of the Vulkan path showed the closest analogue
  uses the Q8_1 integer-dot MMQ route for quantized activations and a
  medium tile for this exact `M=5120,N=64,K=17408` dense-down shape. A local
  low-register MMQ64 approximation did not reproduce the win, so the next
  attempt needs shaderstats or a closer instruction/layout match before
  spending more cycles on dense-down tile shape changes.
- RADV shaderstats on the exact dense-down kernels showed no spills but high
  register pressure: Q6_K BM64 accumulate reported `SGPRs=128`, `VGPRs=228`,
  `LDS=19456`, and Q4_K BM64 accumulate reported `SGPRs=128`, `VGPRs=192`,
  `LDS=16384`. Since the low-register MMQ64 approximation was slower, do not
  assume VGPR reduction alone is enough; preserve exact A/B evidence.
- Single-command-buffer fusion of the SSM/attention prefix plus dense FFN
  was tested and rejected: default-on after warm `148.92,147.78,146.31`
  versus flag-off `149.92,148.25,146.16`; submit boundaries are not the
  limiter for the cold CLI path.
- Prefix-depth zero was tested and rejected: corrected prefix0 samples
  `149.70,149.28,145.12` versus default control `150.94,152.07,147.21`.
- BK4 exact dense-down specialization was tested and rejected: paired
  clean median `149.36` versus BK4 median `148.32`.

2026-06-29 RDNA1 rebooted full-suite notes:

- Rebooted the RDNA1 node before measuring.
- First full run exposed a harness failure mode: Gemma server API responses
  returned without a corresponding ZINC `Generated` timing log, so the
  harness waited for the full remote command timeout.
- Fixed and pushed bounded post-response timing waits in the suite
  (`f719cccc`); missing timing logs now become clear unavailable rows with
  the response/log tail attached.
- Rebooted RDNA1 again and reran the full RDNA suite from clean `f719cccc`.
- Artifact: `/tmp/zinc-rdna1-full-suite-20260629-082140.json`.
- Qwen rows all beat the baseline overall: Qwen 3.6 35B-A3B `152.7%`,
  Qwen 3.6 27B `104.9%`, Qwen 3.5 9B `114.8%`.
- Target `decode-extended` on Qwen 3.6 27B remains ahead:
  ZINC `324.37 tok/s` prefill and `31.64 tok/s` decode versus baseline
  `253.17 tok/s` prefill and `30.40 tok/s` decode.

## Measurement contract

Always measure on a clean RDNA node. Stop stale GPU users first:

```bash
source .env

ssh -p "$ZINC_PORT" "$ZINC_USER@$ZINC_HOST" "\
  pkill -f 'zig-out/bin/zinc' || true; \
  pkill -f 'llama-server' || true; \
  pkill -f 'llama-cli' || true"
```

Full four-scenario comparison:

```bash
source .env

bun tools/performance_suite.mjs \
  --target rdna \
  --models qwen36-27b-q4k-m \
  --phase all \
  --rdna-sync \
  --rdna-build \
  --rdna-start-llama \
  --runs 3 \
  --warmup 1 \
  --timeout-ms 1800000 \
  --no-site-write \
  --output /tmp/zinc-rdna-qwen36-27b-$(date +%Y%m%d-%H%M%S).json
```

ZINC-only cycle check after a small change:

```bash
source .env

bun tools/performance_suite.mjs \
  --target rdna \
  --models qwen36-27b-q4k-m \
  --phase zinc \
  --rdna-sync \
  --rdna-build \
  --runs 3 \
  --warmup 1 \
  --timeout-ms 1800000 \
  --no-site-write \
  --output /tmp/zinc-rdna-qwen36-27b-zinc-only-$(date +%Y%m%d-%H%M%S).json
```

Extract medians from an artifact:

```bash
jq -r '.targets[0].models[0].scenarios[] |
  [.id,
   .zinc.prefill_tps.median,
   .zinc.decode_tps.median,
   (.baseline.prefill_tps.median // "na"),
   (.baseline.decode_tps.median // "na")] | @tsv' \
  /tmp/zinc-rdna-qwen36-27b-ARTIFACT.json
```

Profile a sustained decode prompt:

```bash
source .env

rsync -az --delete \
  --exclude '.zig-cache' --exclude 'zig-out' --exclude 'node_modules' \
  --exclude '.DS_Store' --exclude 'site' \
  -e "ssh -p $ZINC_PORT" . "$ZINC_USER@$ZINC_HOST:/root/zinc/"

ssh -p "$ZINC_PORT" "$ZINC_USER@$ZINC_HOST" "\
  cd /root/zinc && \
  zig build -Doptimize=ReleaseFast && \
  RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
    -m /root/models/Qwen3.6-27B-Q4_K_M.gguf \
    --prompt 'LLM benchmark reports should separate prompt processing from token generation because' \
    -n 48 \
    --profile"
```

Collect these lines every time:

```text
Prefill: ...
Prefill profile: ...
Prefill GPU phases: ...
Prefill SSM subphases totals: ...
Generated ... tok/s
Modeled decode bandwidth: ...
PROFILE: avg GPU decode token=...
PROFILE: avg GPU phases ...
PROFILE: avg SSM subphases ...
PROFILE: avg dense_ffn subphases ...
PROFILE: avg tail subphases ...
```

## Acceptance rules

A performance keep must satisfy all of these:

1. `zig build -Doptimize=ReleaseFast` succeeds on the RDNA node.
2. Local `zig build test` succeeds before commit.
3. No `QueueSubmitFailed`, no GPU reset, no `amdgpu: ring ... timeout`
   in `dmesg`.
4. The output remains coherent on a short raw prompt and the
   decode-extended prompt.
5. For production-on changes, compare the full four-scenario matrix, not
   only one CLI prompt.
6. Keep if median prefill or decode improves materially and no primary
   scenario regresses more than 2%, unless the change is an explicitly
   default-off diagnostic/foundation.
7. For prefill changes, `ZINC_BATCHED_PREFILL=validate` or an equivalent
   new validator must compare final logits against the per-token path.
8. Any SSM prefill change that resets the GPU is rejected even if a
   shorter prompt once passes.

Recommended perf thresholds:

- Prefill production keep: at least +10% on one context scenario and no
  context scenario down more than 2%.
- Decode production keep: at least +3% on the `decode-extended` median
  or at least +2% across two of the four decode medians with no long
  context regression.
- Foundation keep: allowed at 0% only if default-off and it adds
  validation/profiling needed for the structural work.

## Work plan

### Track 0 - Baseline hygiene and profiling

Goal: make every later cycle attributable.

Steps:

1. Run the full matrix and save the artifact under `/tmp`.
2. Run the sustained `--profile -n 48` prompt.
3. Record Mesa/GECC/coopmat state:

```bash
ssh -p "$ZINC_PORT" "$ZINC_USER@$ZINC_HOST" "\
  dpkg -l mesa-vulkan-drivers | tail -1; \
  cat /sys/module/amdgpu/parameters/ras_enable; \
  vulkaninfo --summary | sed -n '1,80p'"
```

4. Confirm no stale processes:

```bash
ssh -p "$ZINC_PORT" "$ZINC_USER@$ZINC_HOST" "\
  pgrep -af 'zinc|llama-server|llama-cli' || true"
```

Deliverable: a short note in the next effort log with:

- artifact path
- ZINC/llama table
- profile phase table
- driver state
- commit SHA

### Track 1 - Prefill validation harness for Qwen 3.6 27B

Goal: build the safety rail before changing production prefill.

Do not start by relaxing `canUseBatchedPrefillRdna`. Instead add a
separate default-off path, for example `ZINC_QWEN36_27B_PREFILL_VALIDATE=1`,
that can run a candidate layer-major calculation and compare it with the
current per-token reference.

The first validator should not attempt to replace the full model output.
It should capture one layer/chunk and compare intermediate tensors:

1. Pick a small token chunk, e.g. 4 or 8 prompt tokens.
2. Run the current per-token path.
3. Capture reference tensors for one layer:
   - pre-FFN `ffn_norm`
   - dense gate output
   - dense up output
   - SwiGLU output
   - dense down output
   - post-FFN hidden
4. Re-run the candidate batched dense FFN for that same layer/chunk into
   scratch buffers.
5. Compare max absolute error and top-k worst indices.

Start with dense FFN because it is stateless across tokens once
`ffn_norm[token]` is known. Do not start with SSM delta recurrence.

Suggested validation tolerances:

- Dense FFN intermediate DMMV outputs: `1e-3` to `3e-3` absolute.
- Final hidden after residual: `3e-3` absolute.
- Final logits after a whole validate run: `1e-3` may be too strict for
  changed reduction order; use intermediate diffs to decide, then set a
  documented threshold.

Useful existing infrastructure:

- `ensureBatchedScratchCapacity`
- `dispatchProjectionBatched`
- `dispatchResidualRmsNorm`
- `dispatchFfnActivation`
- current validate mode in `prefillBatched`
- SSM capture comments/buffers around `runSsmLayerGpu`

Deliverable: default-off validator, no production behavior change.

### Track 2 - Batched dense FFN prefill per layer

Goal: stop rereading the dense FFN weights once per prompt token.

Current per-token dense FFN sequence:

```text
ffn_norm[token]
  -> gate DMMV q4_k
  -> up DMMV q4_k
  -> SwiGLU
  -> down DMMV q6_k
  -> hidden[token] += down
```

Candidate layer-major sequence:

```text
for layer L:
  produce ffn_norm[token] for all tokens in chunk, preserving prior branch output
  batched gate: q4_k [M=17408, K=5120] x [K, N]
  batched up:   q4_k [M=17408, K=5120] x [K, N]
  activation:   swiglu over [N, 17408]
  batched down: q6_k [M=5120, K=17408] x [K, N]
  residual add per token
```

This is the biggest prefill lever because the dense FFN streams about
173 MB of weights per layer per token today. With a chunk of N tokens,
the target is to stream those weights once per layer/chunk instead of N
times. Even after activation traffic and scratch writes, this is the
only route to multi-x prefill movement.

Implementation notes:

- Use existing batched projection helpers first. They may not be the
  final fastest GEMM shape, but they are already wired and safer.
- Keep chunk sizes conservative at first: 4, 8, 16. Do not jump to 128
  until validation passes and scratch memory is audited.
- Store scratch as token-major if that matches existing helpers. Avoid a
  huge layout refactor in the first production attempt.
- Run only one layer in validate mode first, then all layers default-off,
  then production-on after matrix proof.
- If the existing batched DMMV path dispatches one workgroup per
  row/token and underutilizes, pivot to the in-tree `mul_mm_q4k` style
  only after correctness is proven.

Risks:

- A batched dense FFN cannot be run before the layer's attention/SSM
  branch has updated `hidden[token]`.
- For SSM layers, all tokens' branch outputs must be exact before this
  layer's dense FFN can batch.
- Scratch memory can get large:
  - gate/up scratch for 16 tokens: `16 x 17408 x 4` each ~= 1.1 MB each.
  - down/hidden scratch is smaller.
  - This is fine, but keep all buffers sized explicitly.

Acceptance:

- Default-off validator passes for chunks 4, 8, and 16.
- Production-on chunk 8 or 16 improves context prefill by at least 10%.
- Decode does not regress more than noise; the code should not touch
  decode unless helpers are shared.

### Track 3 - Batched SSM projection prefill

Goal: amortize the SSM projection weights while preserving token-order
state recurrence.

Do this only after Track 1 validation exists. The SSM delta/state update
is the risky part; the four projections are less risky:

```text
attn_norm[token]
  -> wqkv q6_k
  -> z q4_k
  -> alpha f32
  -> beta f32
```

Candidate sequence for one SSM layer and token chunk:

1. Produce `attn_norm[token]` for all tokens in the chunk.
2. Batched project `wqkv`, `z`, `alpha`, `beta`.
3. Run conv/delta recurrence in token order or with a separately
   validated batched delta path.
4. Run gated norm and `ssm_out`.
5. Apply residual to each token before dense FFN.

The first production version may still run delta recurrence token by
token. That can still win if SSM projection weight reads are a large
share of prefill. The decode profile says SSM projection is about 5.39
ms of a 12.46 ms SSM bucket, so it is worth isolating.

Do not use the old destructive "skip runSsmLayerGpu and dispatch one
batched delta post-loop" shape. It corrupts later hidden states.

Acceptance:

- Compare `wqkv`, `z`, `alpha`, and `beta` outputs against per-token
  captures.
- Compare `delta_out`, `ssm_out`, and post-SSM hidden after the
  recurrence path.
- No GPU reset on a long-context prompt.
- Context prefill improves beyond Track 2, or the code stays default-off.

### Track 4 - Decode dense FFN, only after prefill foundation

Goal: improve single-token decode without repeating rejected wide/fused
variants.

The direct attempts already say "do not simply widen existing fused
gate/up/SwiGLU". Instead:

1. Add more granular profile counters if needed:
   - separate dense gate q4_k and dense up q4_k, not just `gateup`
   - separate dense down q6_k
   - record dispatch counts per token
2. If shaderstats shows register pressure or occupancy collapse, build a
   shape-specific variant behind an env flag. Candidate:
   - dense Q4_K gate/up/SwiGLU with `NUM_ROWS=1` for `M=17408`,
     designed to reduce register pressure versus the rejected wide fused
     path.
3. Test with full matrix. If it only helps `core`, reject it.

Known-safe decode insight:

- `dispatchDmmvAcc` for dense down already fuses down projection with
  residual add when no post-FFN norm exists. Do not undo that.

Rejected decode idea:

- A one-dispatch `ssm_out + ffn_norm` is not straightforward. `ssm_out`
  is a row-parallel DMMV that writes/accumulates pieces of `hidden`.
  `ffn_norm` requires the RMS of the completed hidden vector. There is
  no global synchronization inside one Vulkan dispatch across all rows.
  A correct version is at least two dispatches or a different work
  decomposition. Treat it as a research item, not a quick fusion.

### Track 5 - Decode SSM

Goal: reduce the 12.46 ms SSM bucket without repeating failed qkv+z and
delta variants.

Open candidates:

1. SSM `out` q5_k row/block tuning.
   - Current profile: `ssm_out ~= 2.20 ms`.
   - Prior wide tests focused on Q4/Q6 FFN and qkv/z, not this q5_k
     output projection shape.
   - Build only if shaderstats shows poor occupancy or memory pattern.
2. SSM projection diagnostics.
   - Split `ssm_proj` timing into wqkv, z, alpha, beta for one profiling
     cycle.
   - If alpha/beta are tiny but high-overhead, revisit fused RMS+alpha+beta
     only as a shape-specific cleanup.
3. Delta-net variants.
   - Current cols8 default is not clearly bad in the full matrix.
   - `ZINC_SSM_DELTA_NORMED_QK=1` regressed; do not retry.
   - Any new delta variant needs a clean state/output diff, not just text.

Acceptance:

- `--profile` shows the target SSM subphase moved.
- Full decode matrix improves, especially `context-long`.
- No prefill regression, because SSM helpers are shared.

### Track 6 - Long-context attention

Goal: address the `context-long` decode gap after dense FFN/SSM.

Attention is only about 2.27 ms on the short sustained profile, but the
long-context scenario has the largest decode deficit. After the dense
and SSM buckets are stabilized, run:

```bash
ZINC_FA_PROFILE_LAYER=1 RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
  -m /root/models/Qwen3.6-27B-Q4_K_M.gguf \
  --prompt "$(cat /tmp/long-context-prompt.txt)" \
  -n 32 \
  --profile
```

Open candidates:

- seq-len-dependent flash attention split/merge parameters, borrowing
  the Effort 11 lesson that long-context and short-context optimal
  dispatch shapes differ.
- KV-cache write fused into K projection on attention layers. Lower ROI
  on 27B than dense FFN/SSM, but structurally safe if kept narrow.

Do not port a broad new attention architecture before proving attention
is a top bucket on the 27B long-context prompt.

## Immediate next-cycle guidance after the 64.87 checkpoint

1. Do not run another blind Q4/Q6 tile-size, barrier, or command-buffer
   fusion attempt. The post-64.87 loop has already measured those as
   flat/dead unless backed by new shaderstats.
2. Collect paired `RADV_DEBUG=shaderstats` for the accepted fused
   Q4_K gate/up/SwiGLU path and accepted tiled Q6_K dense-down path
   before editing either shader again. Record VGPR/SGPR, occupancy,
   spills, LDS, and obvious memory-instruction differences in the
   cycle self-analysis.
3. If dense `gateup` and `down` remain tied, consider SSM projection as
   the next structural bucket. The right version is a validated
   layer-major qkv/z/alpha/beta dataflow, not the old descriptor-offset
   replay path. It must measure flag OFF and flag ON in the same cycle.
4. If benchmark samples are within ~1% of best, require an interleaved
   old-vs-new control or treat the movement as noise. The current keep
   threshold is about `+0.65 tok/s`, so one lucky `65.x` sample is not a
   real win.
5. After any new keep above `65 tok/s`, run the full four-scenario
   matrix before using the result in public benchmarks. The controller
   metric is the Coding Review prefill prompt, but earlier 27B changes
   have helped that prompt while hurting context-long or decode.
6. The controller should count measured-dead rollback cycles as
   no-new-best pressure. They remain valuable information, but a long
   run of measured-dead cycles should trigger pivot prompts and refreshed
   phase budgets instead of printing `stall=0`.

## Next enhancements after the 2026-05-24 post-mortem

The 50-cycle run proved the loop can chase the controller prompt into
local maxima that do not survive a clean build. The next enhancements are
ordered so that trust is re-established before any new perf claim.

### A. Re-establish a trustworthy number (do this first)

1. Run a clean RDNA build + ZINC-only matrix at HEAD `63dc951` using the
   measurement contract above. This is the only way to know what HEAD
   actually does versus the artifact-inflated `79.63`.
2. Decide the fate of the two orphan shaders:
   - If `mul_mm_q6k_full.comp` / `mul_mm_q4k_gate_up_swiglu_full.comp`
     are dead (cyc47 says they regress to `63.71`), delete the `.comp`
     files and the cyc43/cyc44 wiring in `forward.zig`/`dmmv.zig`, then
     re-measure so HEAD matches what it claims.
   - If a corrected full-tile kernel is still wanted, add it to
     `shader_sources`, confirm `mul_mm_*_full.spv` lands in
     `share/zinc/shaders`, and only then measure.
3. Record the corrected baseline in this doc before the next loop starts.

### B. Harness changes (stop the artifact class entirely)

These are the highest-ROI changes; the 27B kernel ideas are worthless
while measurement can silently run stale code.

1. **Shader source/install parity guard. [DONE 2026-05-24]** `buildAndBench`
   in `loops/optimize_perf.ts` now asserts, after `zig build`, that every
   `src/shaders/*.comp` has a matching `.spv` in `zig-out/share/zinc/shaders`,
   and returns `buildOk:false` (error `shader install parity mismatch`)
   listing any that are missing. The audit that motivated this found FOUR
   orphaned shaders never added to `shader_sources`: `dmmv_f32_dual_batch`
   and `ssm_conv1d_batched` (cycle 9) plus `mul_mm_q6k_full` and
   `mul_mm_q4k_gate_up_swiglu_full` (cycles 43/44) — so the entire effort-15
   staircase from cycle 9 on was measured against partially-stale shaders.
   All four were added to `shader_sources` in `build.zig` so HEAD now installs
   exactly what the Zig wiring references; the first re-baseline will report
   the honest number (expect well below the artifact 79.63 — cycle 47 measured
   ~63.71 once the `_full` pair was actually installed). Follow-up still open:
   replacing the hand-maintained `shader_sources` tuple with a `build.zig`
   glob of `src/shaders/*.comp` would make drift structurally impossible.
2. **Variance-aware keep threshold.** The cyc44 samples spanned
   `77.25–80.25` while the keep threshold was `~0.8 tok/s` — inside noise.
   Gate keeps on `n·stddev` of `tokPerSecSamples`, not a flat delta.
3. **Usage-limit backoff.** Detect the agent's "usage limit" error and
   pause/sleep to reset time instead of burning no-op cycles that inflate
   `stall` and trigger false pivots.
4. **Fix or remove `bandwidthUtil`.** It is `0.0` every cycle. For a
   prefill GEMM workload, surface achieved GB/s and ALU occupancy from
   `RADV_DEBUG=shaderstats` instead, so the loop has a roofline signal.

### C. Return to the structural plan, not more micro-fusions

The whole back half of the run was single-shader Q4/Q6 tile/branch
fiddling — exactly what the 64.87 guidance and the failed-attempts table
warned is played out. The unspent lever is still Tracks 1–3: the
default-off layer-major dense-FFN + SSM-projection prefill validator and
batched path. Restart there, with the new parity guard in place so each
keep is real. Do not let the loop spend more than ~3 cycles on isolated
shader micro-fusions before requiring `shaderstats` evidence.

## Suggested cycle schedule

Cycle 1:

- Run baseline matrix and sustained profile.
- Add a short running note with artifact paths and driver state.

Cycle 2:

- Add default-off dense FFN layer/chunk validator.
- No production behavior change.

Cycle 3:

- Validate one layer, chunk 4 and 8.
- Fix diffs until intermediate outputs match.

Cycle 4:

- Extend validator across all layers default-off.
- Confirm no GPU reset.

Cycle 5:

- Enable batched dense FFN prefill for a small chunk behind
  `ZINC_QWEN36_27B_DENSE_PREFILL=1`.
- Measure ZINC-only matrix.

Cycle 6:

- Tune chunk size 4/8/16.
- Keep only if context prefill moves.

Cycle 7:

- Add SSM projection capture/validator.
- No production behavior change.

Cycle 8:

- Batch SSM projections default-off, keep recurrence exact.
- Measure prefill SSM subphase.

Cycle 9:

- If prefill is moving, integrate dense FFN + SSM projection batching
  under one default-off flag and run full all-phase matrix.

Cycle 10:

- Decode-only diagnostic: split dense gate/up/down and SSM projection
  sub-buckets further if the prefill work did not already expose them.

Cycle 11+:

- Try exactly one decode candidate at a time. Prefer q5_k SSM out or
  shape-specific dense gate/up/SwiGLU with lower register pressure. Do
  not mix decode experiments with prefill restructuring in the same keep.

## Files to read before editing

Primary:

- `src/compute/forward.zig`
  - `canUseBatchedPrefillRdna`
  - `prefillBatched` / `prefillBatchedImpl`
  - `prefillBatch`
  - `runSsmLayerGpu`
  - dense FFN path around the `dense_ffn` profile phase
  - `dispatchProjectionBatched`
  - `dispatchDmmvAcc`
- `src/compute/dmmv.zig`
  - batched Q4_K/Q6_K dispatch helpers
  - k-parallel Q4/Q6 paths
- `src/compute/elementwise.zig`
  - residual RMS norm
  - SSM conv/delta/gated norm dispatches
- `src/shaders/dmmv_q4k_batch*.comp`
- `src/shaders/dmmv_q6k_batch*.comp`
- `src/shaders/dmmv_q4k.comp`
- `src/shaders/dmmv_q6k.comp`
- `src/shaders/dmmv_q5k.comp`
- `src/shaders/ssm_delta_net*.comp`
- `src/shaders/residual_rms_norm.comp`
- `src/shaders/rms_norm_add.comp`

Historical context:

- `loops/efforts/MULTI_HOUR_EFFORT_6_RDNA_QWEN36_PREFILL.md`
- `loops/efforts/MULTI_HOUR_EFFORT_8_RDNA_BATCHED_PREFILL.md`
- `loops/efforts/MULTI_HOUR_EFFORT_10_QWEN36_DECODE.md`
- `loops/efforts/MULTI_HOUR_EFFORT_11_RDNA_DECODE_LONG_CONTEXT.md`

### 2026-08-17: baseline corrected (not a regression), SSM prefill-proj batching re-confirmed rejected

**The `llamaCppSuccessRule`/`llamaCppBaselines` "402.92 tok/s" context-medium
figure in this effort's `EFFORT_SPECS` entry does not match this effort's own
benchmark contract and should not be used as the keep/revert threshold.**
Root cause, found by elimination: `optimize_perf.ts` benchmarks by repeatedly
cold-starting the `zinc` CLI (`zincCliArgs` builds a fresh `--prompt`
invocation per sample — full 17 GB model reload each time), while the
published site figure (`tools/performance_suite.mjs`, and the `402.92`
number) comes from one warm, persistent `zinc-server` process hit repeatedly
over HTTP. Ruled out before landing on this explanation: wrong Vulkan device
(`REMOTE_VULKAN_DEVICE_INDEX` already defaults to `1`, confirmed via a fresh
isolated build), shared/contaminated `/root/zinc` build directory (identical
result — 226.11 vs 224.52 tok/s — in a from-scratch isolated directory), GPU
stuck in low-DPM per the known Effort-10 failure mode (`rocm-smi` showed
`Performance Level: high`, `pp_dpm_sclk` at the top `2200Mhz` state), and
prompt-content mismatch (both tools tokenize to exactly 212 tokens, and the
`"You are a helpful assistant. Answer directly. Do not show analysis."`
system prompt `optimize_perf.ts` passes via `--system-prompt` is the same
string the server injects by default for `/v1/chat/completions`,
`src/server/routes.zig:815`). Cold-CLI and warm-server prefill are just
different, both-real metrics; this effort's own contract measures the former.

Corrected current baseline (isolated build, 5 samples,
`ZINC_REMOTE_DIR=/root/zinc-claude-effort15`): **226.11 tok/s** median
`[221.54, 228.03, 224.15, 226.11, 228.77]`. Fresh `ZINC_PREFILL_PROFILE=1`
phase profile at this baseline: `dense_ffn` 375.5 ms total (biggest —
`gateup` 196.8 ms [`gateup_matmul` 194.7 ms: q4 92.9 / q6 101.8], `down`
178.6 ms [`down_matmul` 163.5 ms: q4 73.5 / q6 90.0], `residual_acc` 15.0 ms),
`ssm` 279.4 ms total (`proj` 164.2 ms [`qkv` 130.1 / `norm_ab` 14.3],
`conv` 15.7, `delta` 23.8, `gnorm` 13.5, `out` 61.9 [`out_proj` 48.5 /
`out_resid` 13.3]), `attn` 62.6 ms, `tail` 1.9 ms. Output `"Paris"`,
correct.

Tested the existing `ZINC_QWEN36_27B_SSM_PREFILL_PROJ` flag (the
already-built prebatched-SSM-projection path referenced as this effort's
"unspent structural lever") in all three modes against this corrected
baseline, same prompt, same isolated build:

| mode | prefill tok/s | output |
|---|---:|---|
| off (baseline) | 276.24 (single-sample cross-check; see median above) | `{332, 45209, ...}` |
| `1` (qkv+z) | 211.66 | `{332, 45209, 56692, 332, 198, 760, 9584, 369}` |
| `qkv` | 194.23 | `{332, 45209, 56692, 332, 198, 760, 9584, 369}` |
| `z` | 188.88 | `{332, 45209, 56692, 332, 198, 760, 9584, 369}` |

All three modes are correct (byte-identical output to baseline) but slower —
and mode `1`'s own profile shows the damage is not localized to `qkv`:
`proj` 164.2 -> 195.2 ms, `conv` 15.7 -> 20.4, `delta` 23.8 -> 32.7, `gnorm`
13.5 -> 17.5, `out` 61.9 -> 94.8 ms. Every named SSM sub-phase gets worse,
not just the one the flag directly touches, which points at systemic
overhead from the prebatch mechanism (extra passes/barriers competing with
the main per-layer work) rather than a narrow qkv-dispatch problem. Re-
confirms the flag's existing default-off status is correct for Qwen3.8-27B
specifically, not just inherited unexamined from the 3.6 predecessor. Do
not re-enable without first instrumenting where the prebatch pass's own
cost lands — the buffer-copy-then-dispatch structure at
forward.zig:16835-16878 (a `vkCmdCopyBuffer` per token before the dispatch
that the non-prebatched path doesn't pay) is one plausible source, but the
across-the-board slowdown suggests looking at the upfront batched-
precompute pass itself, not just this per-token copy — needs evidence, not
a rewrite, per this effort's own rule.

RADV shaderstats did not work as invoked (`RADV_DEBUG=shaderstats` on the
main `zinc` binary at runtime produced no VGPR/SGPR/occupancy output) —
`docs/BENCHMARKING.md`'s hot-bench section implies shaderstats needs the
dedicated `zig build hot-bench` target instead, and hot-bench's documented
`--case` list (`q8_router`, `q8_shared_gate_up`, `q8_shared_down`,
`q8_ssm_out`, `ssm_delta`) does not obviously cover the Qwen dense-27B
Q4_K/Q6_K gate-up-down kernels that are this effort's largest bucket. A
real next step is adding a dense-27B-specific hot-bench case (or fixing
shaderstats on the main binary) before spending more cycles guessing at
register pressure blind.

No net speed win landed this session. What's left in good shape: a
corrected, isolated, reproducible baseline and profile: a cleanly ruled-out
false-regression scare (worth remembering — always suspect the harness
before the code when two measurement paths disagree); one more structural
lever (SSM prefill-proj batching) re-confirmed rejected with a concrete
lead (the extra buffer copy) instead of just "still opt-in, unclear why."

## Final decision rule

This effort is successful when one of these is true:

1. ZINC beats llama.cpp on 27B decode in at least three of four
   scenarios and does not lose prefill by more than 20% in any context
   scenario.
2. ZINC improves 27B context prefill by at least 2x over the current
   clean baseline while preserving decode.
3. The effort produces a default-off, validated layer-major SSM+dense
   prefill foundation with no GPU resets and a measured partial prefill
   win, even if full llama.cpp parity needs another effort.

The most likely good outcome is #3 first, then #2. Expecting #1 from
single-token shader tweaks alone is not realistic on this model.
