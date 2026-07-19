# Effort 7 — symmetric Metal perf harness (Metal, M4 Max Mac Studio) — STATUS: OPEN (low priority — motivating symptom resolved elsewhere)

Plan (2026-04-18): measure zinc like llama.cpp — one warm zinc HTTP server per
model via `/v1/completions` — because cold one-shot CLI runs made short-prompt
prefill medians meaningless (first GPU touch faults in ~21 GiB of mmap'd
weights, 3–10 s, published 1.0 vs 2.1 tok/s across otherwise-identical runs).
**None of the plan was implemented** (`launchLocalZincServer` does not exist).
The instability it targeted has since disappeared anyway: with Metal batched
prefill (`prefillBatched`, default-on in CLI and server `routes.zig`)
and load-time weight wiring via `MTLResidencySet`
(`src/model/loader_metal.zig`, `src/metal/shim.m`), the 2026-07-08
suite shows 35B core prefill 97.4 tok/s median, stddev <1% (attribution
plausible, not A/B-verified). What remains is the wall-clock/symmetry argument.

## Landed
- Nothing from this plan. The diagnosis shipped as blog posts only
  (`7c7940eb` cold-CLI prefill noise, `29d6d6fa` MTLBinaryArchive idea).
- Symptom-removing changes landed via other efforts: default batched prefill
  (`src/main.zig:~1868`) and loader residency set (above).

## Dead ends (do not retry)
- None tried — the effort never entered implementation.

## Still open
- Harness asymmetry is still real: zinc = one cold CLI spawn per warmup+run
  (`runSeries` + `localZincCommand`, `tools/performance_suite.mjs`);
  llama.cpp = one warm server per model (`launchLocalLlamaServer` +
  `runOpenAiSeries`). A 6-model suite still pays ~96 zinc model loads —
  the 30–60 min wall-clock saving is unclaimed.
- If implemented: mirror `launchLocalLlamaServer` (port pick, spawn
  `zinc --model-id <id> --port <p>`, wait for health), reuse `runOpenAiSeries`,
  keep CLI fallback on launch failure, keep the JSON schema byte-identical.
  The zinc server mode exists and works (one process per GPU); verify its
  `/v1/completions` exposes prefill/decode timings the OpenAI parser reads
  (llama-server-style `timings` object) (unverified for zinc's HTTP path).
- MTLBinaryArchive PSO caching (the cold-start's other half per the blog) is
  still absent from `src/metal/shim.m` (unverified benefit).

Repro:

```bash
bun tools/performance_suite.mjs --target metal --runs 3 --warmup 1 \
  --models qwen36-35b-a3b-q4k-xl --skip-local-build --no-site-write \
  --output /tmp/zinc-perf-effort7.json
# gate: core zinc prefill stddev/median <= 0.10 across two invocations;
# decode median must not regress; JSON schema unchanged.
```
