# Effort 7: Make the Metal Perf Harness Symmetric (one zinc server per model)

## Current State (2026-04-18)

The shared perf suite at `tools/performance_suite.mjs` measures zinc and
llama.cpp asymmetrically:

- **llama.cpp**: one `llama-server` per model, reused across every scenario
  and every measured run. See `launchLocalLlamaServer` and the comment at
  `tools/performance_suite.mjs:1577`:
  > "llama.cpp is measured from one local llama-server launch per model so
  > the same loaded model can serve the whole scenario matrix."
- **zinc**: `localZincCommand` (line 531) returns a one-shot CLI invocation
  (`./zig-out/bin/zinc -n … --model-id … --prompt …`). `runSeries` spawns
  that CLI fresh for **every** warmup and every measured run. A 6-model
  suite runs zinc ≈ 6 × 2 engines × 4 scenarios × 4 runs = **96 cold
  CLI loads**, vs ≈ 6 cold loads for llama.cpp.

### Why this matters

On Apple Silicon the first GPU access after process start faults in mmap
pages for the model weights (≈ 21 GiB for Qwen3.5-35B-A3B-UD-Q4_K_XL).
That single first-prefill step costs 3–10 seconds and dominates the
measured prefill window on short prompts:

| Scenario                                | Prefill time (1st CLI run) |
|-----------------------------------------|----------------------------|
| `-c 4096 --prompt "What is the capital..."` (12 prompt tokens) | 4.7 s |
| `-c 131072 --prompt "What is the capital..."` (12 prompt tokens) | 10.6 s / 3.4 s / 4.7 s |

The variance is not caused by KV cache size. Repeated back-to-back runs
with `-c 4096` give `2.4 / 2.8 / 3.0 tok/s` prefill; with `-c 131072` the
same three runs give `3.5 / 2.6 / 2.5 tok/s`. The per-token prefill cost
is the same. The variance is the cold-process first-GPU-touch cost, and
it fluctuates with disk cache state, OS page reclamation, and scheduling.

### The visible symptom

The April 18 suite run published `qwen35-35b-a3b-q4k-xl core prefill
median = 1.0 tok/s` vs the April 15 run's `2.1 tok/s`. Both runs are
tight within themselves (stddev 0.09 and 0.28) because they captured
three cold-load snapshots at close points in time. The delta between
runs is not a real zinc regression — it is the suite measuring the same
workload differently depending on OS state.

Meanwhile llama.cpp always reports stable numbers because its server is
warm across the scenario matrix.

## Goal

Measure zinc the same way we measure llama.cpp: **one zinc HTTP server
per model**, reused across every scenario and every measured run, driven
via the same `/v1/completions` path that `runOpenAiSeries` already uses
for llama.cpp.

Side effects:

1. Stable, lower-variance zinc numbers (no cold-load penalty per run).
2. Fair zinc-vs-llama.cpp comparison (both paths warm).
3. ≈ 95 fewer model loads per suite run. Expect 30–60 minutes off the
   Metal suite wall-clock at current scale.
4. Matches how a real user experiences zinc — the server stays loaded.

## Benchmark Contract

This effort is scored on **zinc prefill stability** and **suite
wall-clock**, measured on this local M4 Max Mac Studio.

Primary benchmark command (same one used in the failing-comparison
above):

```bash
bun tools/performance_suite.mjs --target metal --runs 3 --warmup 1 \
  --models qwen35-35b-a3b-q4k-xl --skip-local-build --no-site-write \
  --output /tmp/zinc-perf-effort7.json
```

Keep rules:

1. Zinc `core` prefill stddev must be ≤ 10 % of the median across the
   3 measured runs. (Current observed: 0.09/1.0 = 9 % but with wrong
   median; goal is stability at or above the April 15 range.)
2. Zinc `core` decode median must not regress vs the April 18 run
   (≥ 47 tok/s for Qwen3.5-35B).
3. The produced JSON schema must stay identical — site consumers,
   `site/src/data/zinc-performance.json`, and `tools/print_test_summary.ts`
   do not change.
4. If the server-based zinc path cannot start for any reason, fall back
   to the current one-shot CLI path rather than failing the whole run.

Supporting measurements (run if a cycle looks unstable):

```bash
# Single-model smoke to verify the server path works at all:
./zig-out/bin/zinc --model-id qwen35-35b-a3b-q4k-xl --port 18840 &
sleep 60   # let it finish loading
curl -sS http://127.0.0.1:18840/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen35-35b-a3b-q4k-xl","prompt":"hi","max_tokens":8}'
```

## What The Current Evidence Already Says

1. Zinc already has an HTTP `/v1/completions` path. The suite's
   `runOpenAiSeries` already handles it for llama.cpp and can be reused
   verbatim — only the server-launch helper needs to change.
2. The GPU process lock enforces one zinc per device, so there is no
   concurrent-server ambiguity to design around.
3. The auto-sized KV cache from commit `80ddc6a` allocates ≈ 20 GiB at
   load time; amortising that over all scenarios of a model (instead of
   re-paying it per run) is exactly why this effort pays off.
4. llama.cpp's server launch pattern (`launchLocalLlamaServer`,
   `/v1/completions`) is the reference implementation — copy its shape.

## Working Hypotheses

Treat these as candidates to confirm or refute with measurement:

1. Swapping zinc to a single-server-per-model launch removes the
   cold-load prefill variance for all zinc scenarios on all models.
2. The llama.cpp-style `/v1/completions` path reports prefill and decode
   tok/s symmetric to the zinc CLI's current output. If the HTTP path
   lacks a prefill timing field, the suite must add one (see Step 3).
3. The auto-context commit did not change per-token throughput; the
   suite-observed "regression" disappears as soon as the measurement is
   warm.

## Execution Order

### Step 1: Add `launchLocalZincServer` alongside `launchLocalLlamaServer`

Primary files:

- `tools/performance_suite.mjs`

Tasks:

- Add `launchLocalZincServer(entry, zincBinary, timeoutMs)` that:
  - picks an open port (`pickOpenPort`)
  - spawns `./zig-out/bin/zinc --model-id <id> --port <port>`
  - waits for `/health` to return `{"status":"ok"}` before returning
  - returns the same `{ process, baseUrl }` shape as
    `launchLocalLlamaServer`
- Mirror the error-recovery behaviour: on launch failure, surface the
  captured stderr for diagnosis.

Done when:

- A one-model suite invocation can start a zinc server, reuse it across
  all four scenarios, and tear it down cleanly.

### Step 2: Swap the zinc branch in the per-scenario loop

Primary files:

- `tools/performance_suite.mjs` (the `if (phase.phase === "zinc")`
  branch near line 1458)

Tasks:

- Launch the zinc server once per `entry` (per model) before the
  scenario loop, same structure llama.cpp uses further down in the same
  function.
- Replace the `runSeries(..., command: localZincCommand(...), parser:
  parseZincCliOutput, ...)` call with `runOpenAiSeries(..., baseUrl:
  launchedZincServer.baseUrl, ...)`.
- Keep the fallback path: if `launchLocalZincServer` throws, fall back
  to the current CLI-based `runSeries` so the suite never fails to
  produce numbers for a reachable model.
- Tear the zinc server down at the end of the model's scenarios,
  regardless of success or failure.

Done when:

- A `--target metal` suite run prints `launchLocalZincServer started
  zinc for <model_id>` and all scenarios report numbers that came from
  HTTP, not from `parseZincCliOutput`.

### Step 3: Verify `/v1/completions` reports both prefill and decode tok/s

Primary files:

- `src/server/routes.zig`
- `tools/performance_suite.mjs::parseOpenAiResult` (if renamed, its
  equivalent) — check the fields extracted from the response.

Tasks:

- Confirm that the zinc `/v1/completions` response exposes the prefill
  token count, elapsed prefill ms, and decode elapsed ms. If they are
  missing, add them to the response (non-standard fields are fine — the
  suite is a trusted consumer).
- Extend the suite's OpenAI parser to read those fields when present
  and fall back to `completion_tokens / elapsed_decode_ms` otherwise.

Done when:

- The suite JSON for zinc via HTTP contains `prefill_tps`, `decode_tps`,
  and `total_latency_ms` for every scenario, matching the CLI-era
  schema.

### Step 4: Stability check on the primary benchmark

Primary files:

- `tools/performance_suite.mjs`

Tasks:

- Run `bun tools/performance_suite.mjs --target metal --runs 3 --warmup
  1 --models qwen35-35b-a3b-q4k-xl --skip-local-build --no-site-write
  --output /tmp/zinc-perf-effort7.json`.
- Check that `core.zinc.prefill_tps.stddev / median ≤ 0.10`.
- Run the same command a second time; check that the two runs' medians
  agree within 10 %.

Done when:

- Both stability checks pass for Qwen3.5-35B. Record the median prefill
  and decode numbers in the effort log.

### Step 5: Generalise to all six models

Primary files:

- `tools/performance_suite.mjs`

Tasks:

- Run the full `--target metal --models
  gemma4-12b-q4k-m,gemma4-31b-q4k-m,gpt-oss-20b-q4k-m,qwen3-8b-q4k-m,qwen35-35b-a3b-q4k-xl,qwen36-35b-a3b-q4k-xl`
  suite.
- Confirm wall-clock is ≥ 30 minutes shorter than the pre-effort
  baseline (the April 18 run took 3 h 14 min).
- Confirm that the two Gemma 4 regression cases (0.16–0.20 tok/s) are
  unchanged by this harness change. If they move, that is a separate
  finding — record it but do not gate this effort on fixing them.

Done when:

- Full-suite numbers are published and stable, suite wall-clock has
  dropped, and both the site JSON and `zolotukhin.ai/zinc/` perf table
  can be updated from a single artifact.

### Step 6: Documentation

Primary files:

- `docs/METAL_PERFORMANCE_PLAN.md` or an equivalent harness note.
- `tools/performance_suite.mjs` header doc-comment.

Tasks:

- Note the symmetry change and why. Point readers at
  `launchLocalZincServer` as the warmed-server entry point. State the
  fallback behaviour and when the CLI-era path still runs.

Done when:

- A future maintainer reading the suite source understands why zinc
  runs from a server, not from one-shot CLI.

## Success Criteria

This effort is succeeding when all of these are true:

- `tools/performance_suite.mjs --target metal` launches one zinc server
  per model and runs every scenario from that server.
- Zinc `core` prefill stddev ≤ 10 % of the median for the primary
  benchmark across two independent suite invocations.
- Zinc `core` decode on Qwen3.5-35B is ≥ 47 tok/s (no regression vs
  the April 18 measurement).
- Metal suite wall-clock for the six published models drops by ≥ 30
  minutes.
- The JSON artifact schema is unchanged; site data pipeline keeps
  working without edits outside `tools/performance_suite.mjs`.

## Non-Goals

- Do not change the zinc CLI `--prompt` path. The CLI is still the
  correct measurement for bench-metal microbenchmarks referenced by
  blog posts — this effort only touches the shared perf suite.
- Do not touch the RDNA path. RDNA already runs against a remote node
  with its own launcher logic.
- Do not try to "fix" the Gemma 4 Metal regression in this effort. It
  predates the auto-context change and needs separate analysis.
- Do not change the `/v1/completions` schema in a backwards-incompatible
  way — only additive fields are acceptable.

## Likely Files

- `tools/performance_suite.mjs`
- `src/server/routes.zig` (only if `/v1/completions` needs additive
  fields)
- `docs/METAL_PERFORMANCE_PLAN.md`

## Benchmark Focus

- primary metric: zinc `core` prefill tok/s median and stddev on
  Qwen3.5-35B-A3B-UD-Q4_K_XL
- secondary metric: Metal suite wall-clock for the six published models
- success is judged on stability and wall-clock, not peak tok/s. This
  is a harness cleanup, not a perf optimisation.

## Current Checked-Out Code (build on this code)

- zinc `core` prefill on Qwen3.5-35B: April 18 suite median = 1.0 tok/s
  (samples `[1.2, 1.0, 1.0]`, stddev 0.09) — unstable measurement, not a
  real throughput number.
- zinc `core` decode on Qwen3.5-35B: 47.57 tok/s (stable).
- Metal suite wall-clock for the six published models: ≈ 3 h 14 min.
