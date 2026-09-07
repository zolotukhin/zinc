# Effort 33 — NextN/MTP speculative decoding on the Vulkan backend (R9700, Qwen 3.8 27B) — STATUS: LANDED, 32.9 → 55.5 tok/s SERVER DECODE (+69%), GREEDY OUTPUT BIT-IDENTICAL

Follows [Effort 32](MULTI_HOUR_EFFORT_32_RDNA_QWEN38_27B_DECODE.md) (plain decode
31.3 → 32.1 tok/s on the same card). Same node (`hft`, Radeon AI PRO R9700,
RADV GFX1201, Mesa 25.0.7), same model (`Qwen3.8-27B-Q4_K_M.gguf`, 64 layers =
48 DeltaNet + 16 full attention, one appended NextN block), same suite prompt.

## Result (greedy, 96 generated tokens, server path = benchmark reference)

| path | decode tok/s | ms/tok |
|---|---:|---:|
| Vulkan server, `ZINC_MTP=0` | 32.9 | 30.4 |
| Vulkan server, `ZINC_MTP=1` (default), first landing | 43.2 | 23.2 |
| Vulkan server, `ZINC_MTP=1` (default), after round 2 | 49.9 | 20.1 |
| Vulkan server, `ZINC_MTP=1` (default), after round 3 | **55.5** (55.4–55.7 over 6 runs) | **18.0** |
| Vulkan CLI, `-c 8192`, MTP off / on (round 3) | 32.5 / 54.3 | 30.7 / 18.4 |
| ROCm CLI MTP (earlier work, reference) | 48.9 vs 32.4 | |
| llama.cpp 9400c8946 llama-server, no MTP | 29.9 | |

Acceptance 51–54% with 2 drafts per cycle (2.09 committed tokens per cycle).
Server responses and 400-token CLI generations are byte-identical with MTP on
and off (both stop on EOS at token 255 for the suite prompt). `zig build test
-Dbackend=vulkan`: 617 passed, 1 skipped.

## What MTP does here (mirrors llama.cpp master 465e49b)

- The NextN block runs as a separate "MTP context": input at position p is
  `eh_proj([enorm(embed(x_p)) | hnorm(h_{p-1})])` with `h` = `output_norm` of
  the main model's last hidden state; the block is a full-attention Qwen layer
  with its own KV rows (`n_kv_layers = n_layers + 1`) plus `shared_head_norm`
  and the shared lm head.
- Per cycle: two chained drafts from the NextN block (2.4 ms each, ~540 GB/s
  on the 1.3 GB of NextN + lm-head weights), one **verification pass** of the
  main model over `[seed, draft0, draft1]` (43 ms), accept the matching prefix,
  restore the DeltaNet/conv state to the accepted boundary from a per-layer
  history (0.65 ms), then a NextN catch-up over the committed rows (0.8 ms).
- Prime: after prefill the NextN KV is built over all prompt rows from the
  captured per-row `h` (7–9 ms for 48–67 rows).

## The verification pass: 97 ms → 43 ms (the whole effort in one line)

A 3-token batch through the 64-layer model should cost roughly one decode
token of weight traffic (~27 ms floor) plus per-token recurrence work. The
first working pass cost **97 ms** — no better than three sequential decode
steps — and every probe (kernel choice, activation buffers, barriers, clocks,
VRAM pressure, queue family, driver sync modes, kernel-driver VM flushes and
buffer moves, even byte-level dispatch argument dumps) came back "identical to
decode". Per-layer GPU timestamps finally showed the shape: layers 0–7, 56–63
and every third layer in between took 0.43 ms, the others 2.0 ms. That is
llama.cpp's Q4_K_M layout — those fast layers carry **Q6_K** `ffn_down`, the
slow ones **Q4_K**.

Root cause: `dispatchQwen36DenseDown` takes an early "accumulate into the
residual" branch for Q4_K down projections (`pipeline_mul_mm_q4k_down_acc`, a
32-column prefill GEMM tile) whenever the batched FFN passes the residual
buffer as `accum_target`. It has no lower bound on `n_tokens`, so a 3-token
batch ran full 32-wide tiles: 1.0 ms per layer, ~50 GB/s effective. Q6_K has a
`(n_tokens & 31) == 0` guard in the same block and fell through to the fast
path, which is why only Q4_K layers were slow — and why probes placed in the
fallback path never touched the slow layers. The fix is one condition:
`!self.spec_batch_active` on that branch, so MTP batches take the column DMMV
route like Q6_K. Pass: 97 → 43 ms; MTP: 19.2 → 42.4 tok/s.

Lessons worth keeping: per-layer timestamps beat per-phase aggregates for
"which layers" questions; a probe must be shown to *execute on the slow
instance* before its null result means anything (the log-dispatch probe only
printed Q6_K layers because the budget was exhausted by layers 0–2).

## Other pieces that landed

- **Column-parallel DMMV kernels** `dmmv_q4k_rows4_cols` / `dmmv_q6k_rows4_cols`
  (`ZINC_MTP_COLS=0` disables): 4 rows × up to 4 activation columns per
  workgroup, weights streamed once per matrix. Used for every Q4_K/Q6_K
  projection in a spec batch (gate/up 0.23 ms/layer, down 0.15 ms/layer, lm
  head 2.1 ms). Prefill (≥16 tokens) is untouched.
- **Batched-path gates relaxed only for spec batches** (`spec_batch_active`):
  the layer-major SSM/attention functions accept 2–4 tokens, DP4A tile paths
  are skipped, and the three batched layer functions record into one deferred
  command buffer (`spec_defer_submit`), one submit per pass.
- **Per-token DeltaNet in the batched SSM function** for spec batches: conv +
  delta dispatched per token with the state copied to `state_hist`/`conv_hist`
  after each token so a rejected draft restores in one 3 MB copy per layer.
- **VRAM hold**: `mtpReservedBytes` is now also *held* as a device-local
  buffer from init until `mtpPrepare` runs, so the eagerly allocated KV cache
  cannot push the history buffers into system memory (the auto-context CLI run
  had 333 MB in GTT and 4× slower restores before this).
- **Server integration** (`src/server/runtime.zig` `MtpSource`,
  `src/server/routes.zig`): greedy requests (`temperature`/`top_p`/repetition
  penalty at defaults) whose whole prompt was prefilled in the request use MTP
  cycles as a token source; the streaming and non-streaming loops are
  unchanged otherwise. Vulkan only (comptime-guarded); ROCm keeps its CLI-only
  MTP.
- `ZINC_VK_QUEUE_FAMILY=<n>` (diagnostic): override the Vulkan queue family.
  The R9700 behaves identically on the graphics-capable family 0 and the
  compute-only family 1; it does not expose `VK_KHR_performance_query`, so
  hardware counters are not available on this card through Vulkan.

## Round 2 (target 55 tok/s): 43.2 → 49.9 tok/s

Cycle anatomy after round 2 (CLI, `-c 8192`, 46 cycles for 96 tokens, 2.09
committed tokens per cycle): drafts 3.2 ms, verification pass 38.1 ms, restore
0.6 ms, NextN catch-up 0.7 ms → 42.6 ms per cycle = 20.4 ms/token. Reaching
55 tok/s needs a 38 ms cycle.

Landed (each verified against plain greedy decode: 400-token CLI text and the
server response are byte-identical):

| change | pass ms | tok/s |
|---|---:|---:|
| baseline of round 2 | 42.8 | 42.5 |
| 512-thread `rms_norm_mul_wide` / `residual_rms_norm_wide` for 2–4 row batches (the one-wave-per-row kernels left the GPU idle ~40 µs per dispatch) | 40.2 | 45.0 |
| draft lm-head over the first 131072 rows (`ZINC_MTP_DRAFT_VOCAB`; identical drafts on this prompt, 2.4 → 1.6 ms per draft) | 40.2 | 46.6 |
| `ssm_delta_net_cols8_hist`: one delta dispatch over all tokens writing the per-token state history in-kernel (replaces T dispatches + 3 MB copies per layer) | 39.3 | 47.5 |
| `ssm_conv1d_batched_hist` (same idea for the conv ring) + `dmmv_q5k_rows4_cols` for the Q5_K SSM out projection + rows-8 column kernel for the lm head | 38.1 | 48.9 (server 49.9) |

Rejected on this target (do not repeat without a new hypothesis):

| attempt | result |
|---|---|
| Fused gate+up+SwiGLU column kernel (`dmmv_q4k_fused_gate_up_swiglu_cols`, kept off by default) | neutral: 40.4 vs 40.5 ms |
| Q4_K × Q8_1 dp4a column kernel (`ZINC_MTP_DP4A=1`, off by default) | −1.4 ms only, and the quantized activations flip a near-tie token (text differs at byte 354) — not exact |
| rows-per-workgroup 2 / 8 for the f32 column kernels (`ZINC_MTP_COLS_ROWS`) | 41.9 / 41.7 vs 40.4 for rows 4 |
| 16-byte header loads in the Q4_K column kernel | no change |
| K-split 2 / 4 with a fixed-order reduce for the Q4_K column kernels | 41.5 / 46.2 vs 38.3 (the extra reduce dispatch+barrier per matrix costs more than any wave-balance gain) |
| 3 drafts per cycle (`ZINC_MTP_DRAFTS=3`) | 2.17 tok/cycle but the T=4 pass costs +4 ms: 43.4 vs 44.5 tok/s |
| specialization constants for NUM_ROWS in the column kernels | glslang/ACO produce garbage (array sizes fold to the default while the row mapping specializes); use separate shader files |

Why the pass stalls at ~38 ms: profiled per pass, gate/up 13.5 ms (0.21
ms/layer vs 0.17 in decode), down 9.3 (0.145 vs 0.11), SSM projections 6.3,
delta 2.0, SSM out 2.3, attention 3.2, tail ~3. A diagnostic column kernel
with all activation loads removed still ran gate/up at 13.1 ms, i.e. the
rows4-cols kernel's *weight* side streams at ~490 GB/s where decode's single-row
fused kernel reaches 590 GB/s, and neither rows, header-load width, K-split,
dp4a nor fusion moved it. Closing the remaining ~5 ms needs a different
small-batch matvec design (LDS-staged activations with a decode-shaped weight
stream, or a wave32 variant); the other ~2 ms sit in the SSM projection column
kernels for the same reason. With that, the cycle would be ~36 ms and the
target reachable.

## Round 3 (target "55 stable"): 49.9 → 55.5 tok/s

Cycle after round 3 (CLI, `-c 8192`): drafts 2.75 ms, verification pass 34.6
ms, restore+catch-up 0.95 ms → ~38.3 ms per cycle at 2.09 committed tokens.

| change | pass ms | CLI tok/s |
|---|---:|---:|
| start of round 3 | 38.1 | 48.9 |
| **unguarded 3-column kernels** (`dmmv_*_rows4_cols3`, lm head `dmmv_q6k_rows8_cols3`): the per-row `if (row >= M) break` inside the unrolled row loop and the `if (c < num_cols)` around the activation loads kept the compiler from issuing an iteration's 24 loads up front; our shapes are all multiples of 4 and a 3-token batch has exactly 3 columns, so both guards go | 35.0 | 52.5 |
| draft: block input + pending-h upload + NextN step in one command buffer; restore copies + catch-up in one; the pass records its own embedding copy (12 → 4 submits per cycle) | 35.0 | 53.0 |
| K/V-only NextN catch-up (`spec_kv_only`: no Q/gate projections, flash, o-proj or FFN-norm tail; also prime 9.0 → 6.8 ms) | 34.7 | 53.7 |
| draft lm-head over 98304 rows (server 54.4 → 54.8) and the unguarded fused gate+up+SwiGLU 3-column kernel (`ZINC_MTP_FUSED_GATEUP`, now on) | 34.6 | 54.3 (server 55.5) |

Neutral or rejected this round: process-wide `RADV_PERFTEST=cswave32` (one
36.3 ms reading that never reproduced; explicit wave32 on the column
pipelines is 3 ms *worse*, `ZINC_MTP_COLS_WAVE32` keeps the driver default),
rows 8 for gate/up (`ZINC_MTP_WIDE_ROWS`), serializing gate and up
(`ZINC_MTP_GATEUP_SERIAL`, +0.4 ms), compute-only layer hand-off barriers
(no change, kept), the fused norm+RoPE+KV kernel (not applicable: Qwen 3.8
uses precomputed IMROPE frequencies).

The pass is now ~34.6 ms for 3 tokens vs ~31 ms of decode-rate streaming;
the FFN column kernels stream at ~500 GB/s (decode 590) with activation
re-reads accounting for most of the rest. Acceptance is prompt-dependent
(47–54% on this prompt family), so "stable" here means the benchmark prompt
and the server path; longer generations with lower acceptance land in the
low 50s.

## Validation with MTP enabled (2026-09-07, node `hft`, `ZINC_MTP` default on)

- `zig build test -Dbackend=vulkan`: 617 passed, 1 skipped.
- `bun test` on the node (Vulkan build, node model files via
  `ZINC_QWEN3_8B_MODEL` / `ZINC_QWEN36_35B_MODEL` / `ZINC_QWEN38_27B_MODEL`):
  638 passed, 3 skipped, 1 failed. The three CLI smokes (Qwen3.5 9B, Qwen3.6
  35B-A3B, Qwen3.8 27B: first token 11751, "Paris") pass through the MTP
  generate loop. The failure was `source_hygiene` (a comment naming the
  comparison runtime), fixed in `b0de412c`. The skips are the managed-cache
  chat smokes (the test looks in the macOS cache path) and the API smoke,
  which needs a live server URL.
- Against a live Qwen 3.8 27B server (`ZINC_API_BASE_URL=http://127.0.0.1:<port>/v1`):
  `OpenAI API smoke > external server` passes and
  `bun tests/test_openai_sdk.ts --base-url ...` passes all 13 checks
  (non-streaming, streaming, sequential and overlapped streams, health under
  load, completions, error format, 404, OpenAI SDK streaming and
  non-streaming). Gotcha: the CLI smokes cannot run while a server holds the
  GPU (the process lock makes them wait out their timeout), so run the
  server-facing tests separately.

## Round 4 (2026-09-07, afternoon): keep the suite's core row at 55

Goal: the published core scenario (96 greedy tokens, chat prompt, CLI with
auto context) read 52.9 while the server A/B read 55.5; keep 55+ everywhere.

1. **The CLI's auto context spilled VRAM.** The Vulkan CLI handed the engine
   the architectural maximum (262144) and let it trim against the *full* card:
   30851 tokens, 32.4 GB of 32.6 GB used, ~300 MB evicted to system memory.
   Decode paid 17% with MTP (44.8 vs 54.0 tok/s at `-c 16384`) and 10% without
   (28.9 vs 32.0). Plain decode with an explicit `-c 30851` was fine only
   because it has no MTP history buffers. Fix: the CLI now plans the context
   like the server and the Metal CLI (`memory_plan.autoContextTokensForDeviceBudget`,
   85% of VRAM): 22016 tokens, no spill, 53.7 tok/s on the suite command.
   The per-token flat KV cache is allocated for all 65 layers (48 of them
   DeltaNet layers that never read it: 532 KB/token); trimming that is a
   separate follow-up worth 4x the context.
2. **Chained cycle.** The two drafts and the verification batch are one
   command buffer; the draft token never leaves the device. A new
   `embed_gather` shader dequantizes the embedding row of a device-side token
   id (Q4_K/Q6_K/Q8_0/F16/F32, `precise` so it is bit-identical to the host
   `dequantRow`) into the next draft's block input and the verification rows.
   `decodeStep` gained `decode_defer_submit` (record only), `mtpDraftStep`
   split into record/sample halves, `mtpTargetBatch` takes rows 1.. as
   pre-gathered. Worth ~0.3 ms per cycle (the round trips were cheaper than
   estimated: the drafts' 2.7 ms wall was mostly GPU work).
3. **Deferred restore + catch-up.** The rejected-draft state restore and the
   NextN catch-up (rows 1.. only; row 0 duplicates what draft 1 wrote) are
   recorded at the start of the next cycle's command buffer instead of their
   own submit. Anything else that touches the state flushes first
   (`mtpFlushPending` in plain `decodeStep`, both prefill entries; a new
   request discards it). Catch-up wall 0.9 → 0.1 ms per cycle.
   Bug found on the way: the deferred catch-up staged its embedding rows in
   `prime_embed_staging`, which the verification batch then overwrote with
   the seed row before the buffer was submitted, so position pos+1 of the
   NextN cache saw the wrong token — the server's acceptance dropped from
   51% to 47% while the CLI prompt happened to be unaffected. The batch now
   copies the seed row from `embed_staging` (draft 1's own row).
4. **Asynchronous catch-up (tried, reverted).** Submitting the catch-up
   without waiting and letting the next cycle queue behind it moved the same
   GPU time into the next wait; neutral, more fences — removed.
5. **Three drafts.** Six four-column kernels (`*_cols4`, MAX_COLS = 4 siblings
   of the cols3 ones) make the 4-row pass 42 ms instead of 48, but it is
   still ~8% dearer than the 3-row pass and the third position accepts far
   less than the second (p3 ≈ 0.25 vs p2 ≈ 0.8 on context-long), so fixed
   three drafts lose (53.7 vs 54.2) and the adaptive policy
   (`ZINC_MTP_DRAFTS=auto`, EMA per-position acceptance, switch when
   p1·p2·p3 > 0.13·E2) is a wash: context-medium +2.2, context-long −1.5,
   core −0.2. Default stays 2; both remain opt-in.
6. **Draft vocabulary.** 81920 rows: +0.3 tok/s on the core prompt, but loses
   drafts on context-long and decode-extended (−0.8, −1.5); 65536 worse.
   Default stays 98304.
7. **Server.** The draft state is now prepared before the first prefill (it
   was prepared lazily in `mtpPrime`, so request 1 of every server ran plain
   decode); the server logs per-request MTP statistics when the next request
   begins.

Results (R9700, greedy, byte-identical to plain decode on every prompt):

| scenario | before (suite, 2026-09-07 morning) | after |
|---|---:|---:|
| core, 96 tokens (CLI auto ctx) | 52.9 (45–48 when the spill hit) | 54.6 |
| core, 97 / 255 tokens | — | 55.2 / 54.1 |
| context-medium, 160 | 68.5* | 60.2 |
| context-long, 128 | 58.6 | 66.2 |
| decode-extended, 256 | 61.2 | 66.3 |
| server, core prompt, 96 tokens | 55.5 (round 3) / 51.3 (chained, before the staging fix) | 53.6 |

(*) the morning suite ran with the spilling planner; its per-scenario numbers
were partly luck (VRAM 99.4% full, evictions not deterministic).

Per cycle now: verify ≈ 34.8 ms, drafts ≈ 2.5 ms, restore/catch-up ≈ 0.5 ms,
overhead ≈ 0.3 ms; 2.09 tokens per cycle on the core prompt. The verify pass
is at ~80% of the 28.3 ms weight-streaming floor; the rest is ~800 small
dispatches (the 48 DeltaNet layers run ~10 each). Fusing the SSM layer's
z-projection/gated-norm/out-projection chain is the next real lever.

## Round 5 (2026-09-07, evening): the benchmark harness was pinning the clocks down

The refreshed suite still published 53.6 tok/s for the core scenario while the
same command run by hand gave 55.9. Two findings, both about the harness rather
than the engine:

1. **The RDNA target measures the server, not the CLI.** `runRdnaTarget`
   launches `zinc --port N` and drives it through the OpenAI API, so every
   published prompt carries 19 extra chat-template tokens (core prefill 67, not
   48). Per *cycle* the CLI and the server are identical (38.2 vs 38.1 ms); the
   published number is lower only because that longer prompt accepts 51.1% of
   drafts instead of 54.3%. Nothing to fix, but per-scenario CLI numbers are
   not comparable to the published ones.
2. **The suite forced `power_dpm_force_performance_level=high` before every
   run, and on the R9700 that is 4.3% slower than the card's default `auto`.**
   Same binary, same prompt, same 48/94 acceptance:

   | DPM | server decode |
   |---|---:|
   | `high` (what the harness pinned) | 53.5 / 53.6 / 53.6 |
   | `auto` (card default) | 55.91 / 56.14 / 56.00 / 55.94 |

   Forcing `high` locks a fixed nominal DPM state and gives up opportunistic
   boost. The pin came from the 9070 XT work (blog post
   `2026-06-14-rx-9070-xt-prefill-mystery-25x-regression-mesa-devel`), where
   `auto` let SCLK fall to an idle 0 MHz state and consecutive samples ramped;
   on this card `auto` is as stable as `high` (0.4% spread over four runs) and
   the suite already runs a warmup before the measured runs. `rdnaDpmHighScript`
   is now `rdnaDpmNormalizeScript` and sets `auto` (PCIe ASPM still forced to
   `performance`). Both runtimes are measured under the same policy, so the
   comparison stays fair — llama.cpp's own numbers move with it.

## Known gaps / follow-ups

- **Cached-prefix sessions**: when the server reuses a prompt prefix
  (`session_id` clients), the NextN KV for the reused rows and the previous
  row's `h` are not primed, so those requests fall back to plain greedy decode.
  Fix sketch: keep the NextN KV complete for the whole transcript (one extra
  catch-up over the final token at request end), keep a host copy of `h` per
  transcript row, and prime only the suffix rows.
- **Sampling**: MTP is greedy-only (as in llama.cpp's initial MTP); requests
  with temperature > 0 use the ordinary path.
- **CLI auto-context**: fixed in round 4 (the CLI plans with the 85% rule
  like the server). The flat KV cache still allocates 532 KB/token for all 65
  layers although only 17 attend; sizing it for the attention layers would
  quadruple the auto context of this hybrid.
- **Remaining pass headroom (~7 ms of 38)**: see "Why the pass stalls" above;
  small attention dispatches for 3 rows (~0.7 ms) and the three sequential
  argmax reductions in the tail (~0.2 ms) are the only cheap leftovers.
- `ZINC_MTP_DRAFTS` (1–3) is exposed but only 2 was tuned on this card.

## Measurement notes

- `/root/mtp_probe.sh "<label>|ENV=..."` prints per-cycle pass cost from the
  CLI; `/root/server_ab.sh <workdir> <port> 3` for the server path (start the
  next server only after the previous one has exited, or `/v1/models` returns
  an empty list and every request 404s).
- `ZINC_MTP_PROFILE=1 --profile` keeps one submit per layer function so the
  `--profile` phase totals attribute GPU time ("NextN/MTP GPU phase totals").
- DPM on the node reads `high` after every perf-suite run: the suite's remote
  command prelude (`rdnaDpmHighScript`) pins it and does not restore it. Clocks (SCLK ~2290, MCLK 1258) were
  identical for decode and for the slow pass, so clocks were never the cause.
