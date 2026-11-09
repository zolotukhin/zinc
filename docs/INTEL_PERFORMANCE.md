# Intel Arc performance qualification

The target is ZINC Vulkan against current llama.cpp Vulkan and native
SYCL/Level Zero on the same Intel Arc GPU. These are separate campaigns.
Both engines must offload the model to that GPU. There is no CPU-only inference
baseline. CPU unit tests and numerical shader references remain correctness
checks, separate from performance comparisons.

[`tools/benchmark_intel.py`](../tools/benchmark_intel.py) runs on the remote Linux
worker. It covers all seven pinned Intel release models with the existing four
workloads: short chat, medium context, long context, and extended decode. It first
runs 376 numerical cases, tokenizer parity, complete native-reference
continuations, and API cancellation/error recovery. Gemma also crosses its
sliding-window boundary. Correctness uses the release policy's reference pin;
performance uses the independently refreshed latest llama.cpp pin.

For each model, both reusable servers use the same GGUF, context limit, F16 KV,
one slot, disabled MTP and prompt caching, and identical request payloads. The
default three rounds alternate engine order. Each workload gets one warmup and
three measured requests per round. Natural early EOS remains in the report and
excludes that workload from full-budget comparisons; finishing early is never
counted as a speed win. Reduced prefill work, different prompt counts, invalid
token accounting, and unstable repeated outputs fail the campaign.
Workload messages are rendered once by the frozen correctness oracle. Both
servers receive identical raw prompt bytes through `/v1/completions`, including
the separate streaming probes. Development ZINC builds with raw SSE support
are required; the 0.1.0 Vulkan archives do not implement raw streaming.
llama.cpp must report full GPU layer offload; ZINC must confirm F16 KV storage.

Run the following **on the remote Intel worker**, with paths supplied by private
operator configuration:

```bash
python3 tools/benchmark_intel.py \
  --zinc-tree "$ZINC_PACKAGE_TREE" --zinc-source "$ZINC_SOURCE" \
  --llama-bin "$LLAMA_VULKAN_BIN" --llama-source "$LLAMA_SOURCE" \
  --oracle "$ZINC_REFERENCE_BIN" --oracle-source "$ZINC_REFERENCE_SOURCE" \
  --worker-config "$ZINC_WORKER_CONFIG" --output "$ZINC_BENCHMARK_REPORT"
```

Refresh and build the performance reference before starting, following
[the benchmark guide](BENCHMARKING.md#latest-llamacpp-and-matched-mtp-pairs).
The harness holds the shared worker reservation, refuses existing inference
processes, and reaps each owned server group before starting another. Keep the
SSH invocation alive until it finishes.

The report retains every measured request's prefill/decode rates and full HTTP
latency. A separate warmed 16-token SSE probe measures time to first usable
content for each workload and round. Startup measures time until health is
ready, with OS caches retained; it is not a cold-disk claim. Health is polled
every 10 ms in new campaigns, so sub-10 ms startup differences are unresolved.
Historical 500 ms samples retain their original resolution. The measurement
includes health-request latency and the wait between polls; it remains observed
HTTP readiness with retained OS caches. Each server records its polling interval,
and a comparison uses the coarsest interval across both engines and all rounds.
Missing historical per-server metadata defaults to 500 ms. Cross-protocol data
cannot inherit the newer run's finer resolution. In these timing campaigns a
successful JSON response counts as ready only when its health status is `ok`;
loading and malformed replies keep polling within the six-minute startup deadline.
Other release-helper callers retain their existing response contract and interval.
Linux process status
and DRM fdinfo supply host RSS and observed GPU resident/allocated memory peaks.
GPU peaks are sampled every 50 ms, and unavailable counters remain unknown.
Client identifiers and PCI addresses are never published. The counter semantics
come from the [Linux DRM usage statistics documentation](https://docs.kernel.org/next/gpu/drm-usage-stats.html).

A subset selected with `--models`, or fewer repetitions, is a diagnostic. Passing
means that correctness and measurement contracts passed; it does not mean ZINC
beat llama.cpp. `comparison_eligible` identifies workloads with complete output
budgets on both engines; `complete_performance_matrix` requires them for all
seven models. Review all model/workload results, retained bytes and settings
before publishing comparisons. This single-slot protocol also does not establish
concurrent-client capacity or MTP-on performance; those need separate matched
campaigns. Release candidate qualification remains a separate gate on the final
packaged bytes.

Each model now includes `performance_comparison`: prefill and decode throughput,
HTTP latency and first usable streaming content for every workload, plus startup,
host peak RSS and observed resident/allocated VRAM. Each dimension retains the
paired medians for every round. A losing round cannot disappear inside a pooled
median, and unavailable counters remain unknown. Startup differences within the
recorded polling resolution are unresolved. Ties are not wins.
Each round records its expected, observed and valid sample counts so missing,
extra and invalid measurements can be distinguished. `measured_dimensions`
counts dimensions with complete paired data across every round;
`headline_eligible` also requires all dimensions to be measured.

`all_models_all_dimensions_better` requires the complete seven-model matrix,
all four full-budget workloads, at least three rounds with three measurements
per engine/workload, and improvement in every measured dimension in every round.
The matched single-slot F16 KV, MTP-off and disabled-cache settings must also
hold. The existing `passed` field records correctness and measurement validity;
it can be true while the performance objective remains false. A diagnostic
subset, an early-EOS workload, an unknown memory counter or a faster pooled
median cannot establish the full objective. Repeated observed improvements do
not themselves establish statistical confidence or concurrent-client capacity.

Model coverage is derived from completed results, including successful native
inference checks, rather than the requested model list. Missing, duplicated or
unfinished model rows cannot establish a complete matrix. The catalog and Intel
policy are checked together in CI so a new managed model cannot silently escape
qualification. A CPU-only Zig exporter reads the compiled catalog data, without
source-text parsing or GPU initialization. Each measured baseline server must retain evidence of complete
Vulkan GPU offload, and each ZINC server must confirm F16 KV. Cached comparison
flags are refreshed from the samples before deriving the campaign result. Extra
or restarted servers remain in the evidence with an explicit ineligibility
reason; warmup requests reuse the measured server. An all-model win also requires the
recorded upstream HEAD check to match the performance reference and the hardware
identity to remain unchanged through cleanup.
These stricter summary gates apply to new campaigns; historical reports retain
their recorded fields and measurements.

New campaigns probe GPU, kernel and Vulkan driver identities again after all
servers have stopped. A changed identity fails the campaign; `hardware_unchanged`
records the successful end check. Stack upgrades belong between campaigns.

The [pipeline-loading investigation](VULKAN_PIPELINE_LOADING.md) records a
controlled initialization A/B, including its separate driver-cache experiment
and the unchanged inference throughput.

## Recorded diagnostic and next bottlenecks

The [November 5 seven-model GPU diagnostic](/zinc/validation/intel-vulkan-performance-diagnostic-2026-11-05.json)
uses ZINC built from `c56d2888`, performance reference `8bed7cfb` and correctness
oracle `5f436ddd` on Intel BMG G31 with Mesa 26.2.3. Both inference engines use
Vulkan and fully offload the model to the same GPU. One round with one warmup
and one measured request per workload establishes directions for optimization;
it is not the default three-round headline campaign.

Short-chat rates below are tokens per second. The linked record retains all
four workloads, exact prompts, output counts, source and archive identities,
memory samples, startup and streaming latency.

| Model            | ZINC prefill | llama.cpp prefill | ZINC decode | llama.cpp decode |
| ---------------- | ------------ | ----------------- | ----------- | ---------------- |
| Qwen 3.5 9B      | 384.9        | 182.0             | 63.9        | 68.1             |
| Gemma 26B        | 66.6         | 217.3             | 63.8        | 81.2             |
| Gemma 31B        | 52.8         | 47.7              | 19.6        | 21.9             |
| Qwen 3.6 35B A3B | 99.7         | 190.7             | 87.1        | 89.6             |
| Qwen 3.6 27B     | 143.7        | 45.4              | 21.5        | 24.3             |
| Qwen 3.8 27B     | 147.1        | 45.3              | 21.5        | 24.2             |
| Qwen 3 0.6B      | 246.1        | 2531.9            | 193.2       | 292.6            |

The GPU, kernel and driver identities stayed unchanged throughout the run. The
376 numerical cases pass. All seven models' tokenizer, complete native
continuations and API recovery controls pass: 431 of 432 CLI choices are exact,
with one qualified near tie on Qwen 35B. Both Gemma models cross their sliding
windows. The [November 3 diagnostic](/zinc/validation/intel-vulkan-performance-diagnostic-2026-11-03.json)
remains frozen as historical evidence.
The small starter naturally reaches EOS in the long-context
workload, so that pair is retained but excluded from full-budget comparisons.
The report therefore has a complete model matrix and an incomplete performance
matrix. It does not establish that every model is suitable for coding tasks.

Decode remains behind the native GPU reference on every model. Prefill is mixed:
Gemma 26B's conservative serial path and the small starter are especially large
gaps. Qualifying faster paths requires complete native continuations and recovery
checks; isolated shader math agreement alone is insufficient. After a change
passes those controls, repeat the matched three-round campaign before claiming
a performance win.

### Q8 starter prefill investigation

The [targeted Q8 investigation](/zinc/validation/intel-q8-prefill-investigation-2026-11-05.json)
compares packaged baseline `c56d2888` with the opt-in tiled path in `1a4c2fae`,
using the same model, GPU, F16 KV and MTP-off requests. In one round with three
measured requests, short-chat prefill rises from 246.8 to 1,908.7 tokens/s;
decode stays near 193 tokens/s. llama.cpp `8bed7cfb` remains ahead at
2,676.6 prefill and 287.0 decode tokens/s in that diagnostic.

The tiled path passes all eleven F32-KV full-vocabulary checks with a maximum
absolute difference of 0.000061. Its separate F16 native checks include three
short and three extended prompts: 95/96 choices match exactly, with one
qualified near tie. All eighteen complete HTTP continuations pass, including
replays after cancellation and context errors. The old eligibility gate fails
the numerical fixture by silently falling back to serial prefill; that negative
control is retained in the report. The guarded default now requires Intel,
the complete all-Q8 graph and its loaded F32-activation kernel. Setting
`ZINC_INTEL_BATCHED_PREFILL=0` retains the serial control.

A separate experimental decode shader reaches 227.9 tokens/s in that same
short-chat diagnostic and passes its numerical/native controls. It is not part
of the prefill default. The report retains that experiment's GLSL source,
native-quantizer fixture and all treatments. Neither experiment establishes
the three-round objective, and the starter still ends early on context-long.

The [batch-size and decode follow-up](/zinc/validation/intel-q8-tuning-2026-11-05.json)
keeps that decode control while testing 96-token, 384-token and monolithic
prefill. The bounded 384-token path improves medium-context prefill from
1,799.5 to 2,756.3 tokens/s and long-context prefill from 1,909.7 to 3,575.5.
The long-context output still stops early and does not establish a full-budget
win. Three native prompts beyond 1,100 tokens and all eighteen HTTP
continuations pass for the larger batch, including cancellation/error replay.
The Q8 default therefore uses 384 tokens; Gemma retains its independent tuning.
The four-row decode layout adds no material gain in this diagnostic and remains
an experiment.

The [final packaged-default check](/zinc/validation/intel-q8-packaged-default-2026-11-05.json)
tests the merged code without an experimental flag. Three rounds, each with
one warmup and three measured requests per workload, retain 144 requests and
36 streaming probes. Short-chat prefill improves from 246.31 to 1,912.45
tokens/s, with HTTP latency falling from 729.20 to 528.00 ms. Decode stays at
193.20 tokens/s. The newly built llama.cpp Vulkan reference, selected at
`7d516e25` before that campaign, reaches 2,697.71 prefill and 288.15 decode
tokens/s on the same workload. The starter's context-long output still ends
early on both engines; the full performance objective remains unmet.

All fourteen F32-KV full-vocabulary checks pass at the unchanged 0.001
tolerance, with maximum difference 0.000061. Actual F16-KV inference matches
all 96 native choices across short and >1,100-token prompts. The same package
passes 376 native-reference kernel cases, eighteen full HTTP continuations
with recovery, and the explicit serial-off control. Three active-prefill
cancellations at 16K context recover in 2.21–2.23 seconds; each is followed by
six complete native HTTP replays. The report retains the earlier unexercised
short-context cancellation probe, exact artifact identities and raw samples.
This validates the Q8 default; it does not replace a complete all-model campaign.

The [seven-model packaged replay](/zinc/validation/intel-packaged-seven-model-replay-2026-11-05.json)
separately runs that package on every qualified model. It reproduces all 432
frozen qualified choices (431 primary native choices and one already qualified
near-tie branch), plus 81 complete HTTP continuations before and after recovery.
Both Gemma window sets, advertised context boundaries, chat streaming and
Unicode checks pass. This reuses the complete frozen native oracle results;
it is a fresh package replay, with the existing OpenCode coding results kept
separate. Hardware and package identities remain unchanged through the run.

The [bounded-upload investigation](VULKAN_UPLOAD_STAGING.md) separately records
the temporary-memory reduction and loading-time observations. Its A/B compares
two ZINC packages and does not establish a speed advantage over llama.cpp.

### Q8 decode candidate: correctness and controlled timing

The [November 7 development-candidate investigation](/zinc/validation/intel-q8-decode-investigation-2026-11-07.json)
records the SIMD32 row-layout candidate in [PR #75](https://github.com/zolotukhin/zinc/pull/75), built with Zig 0.17.1 at
`51dd99dd` (tree-identical to PR head `520b46e9`). It is separate from the
published 0.1.0 archives. All seven models pass the resumed native/API replay:
431 of 432 choices match the primary native sequence, with one already qualified
near-tie branch. All 81 full HTTP continuations, both Gemma sliding-window sets,
context boundaries, cancellation/error recovery, chat streaming and Unicode pass.
The interrupted reports remain preserved. A fresh post-reboot run also passes
376 base numerical cases and the expanded Q8 fixture's 180 cases / 90 bitwise
pairs, with the package, shaders, oracle, kernel and Mesa identities unchanged.

Three balanced-order rounds compare the same starter-model GGUF through reusable
servers: the packaged legacy override, the packaged candidate default and
freshly built llama.cpp Vulkan `9f0e5098`. F16 KV, MTP off, one slot and disabled
prompt caching match throughout. The record retains 144 requests, 36 streaming
probes and nine server lifetimes; candidate and legacy complete outputs match.
Short-chat round-median decode rises from 192.94 to 250.91 tokens/s, about 30%,
and HTTP latency falls from 529.16 to 412.78 ms. The native competitor remains
ahead at 288.56 decode and 2,697.46 prefill tokens/s; candidate prefill is 1,978.22.

This remains a one-model performance diagnostic. Natural EOS on context-long
is retained and excluded from full-budget comparisons. Startup in this run used
500 ms polling and remains unresolved at that resolution; the newer 10 ms
measurement protocol does not relabel these samples. Actual OpenCode/OpenClaw
acceptance and broader performance checks are separate gates before promoting
the candidate. These results do not establish an all-model performance win.

### Broad row-layout default rejected after all-model controls

The [expanded numerical and six-model timing follow-up](https://zolotukhin.ai/zinc/validation/intel-q8-decode-qualified-followup-2026-11-07.json)
retains the original candidate archive and every round. The Q8 numerical fixture
now covers every managed Q8 input width: **228 native-reference cases and 114
bitwise pairs** pass, including shared-expert, Gemma and NextN widths skipped by
some ordinary MTP-off replays. Correctness does not establish a speed benefit.

Three alternating-order rounds on each of the six larger models compare the
packaged legacy and broad row-layout paths through reusable servers. All 144
complete responses match byte for byte. Core decode changes are:

| Model            | Candidate versus legacy, each round |
| ---------------- | ----------------------------------- |
| Qwen 3.6 35B-A3B | +2.33% to +2.36%                    |
| Qwen 3.6 27B     | -0.05% to +0.09%                    |
| Qwen 3.5 9B      | -0.47% to -0.33%                    |
| Gemma 4 31B      | -0.20% to 0.00%                     |
| Gemma 4 26B-A4B  | -4.59% to -4.56%                    |
| Qwen 3.8 27B     | -0.09% to +0.09%                    |

**The broad default is not promoted.** The revised default requires the dense
Qwen2/3 graph and Q8_0 on every loaded matrix, including the embedding/LM head.
One-dimensional norm scales do not affect that classification. Mixed
quantization, MoE, recurrent and sliding-window graphs retain the legacy path.
`ZINC_INTEL_Q8_DECODE=1` remains an explicit broader experiment;
`ZINC_INTEL_Q8_DECODE=0` is the legacy control. Unknown settings disable the
candidate. SIMD32 capability checks and missing-pipeline fallback still apply.
The fresh package checks below confirm that selector; the historical archive
results are not relabeled as measurements of the narrower build.

The [client matrix](INTEL_OPENCODE_VALIDATION.md) separately records real coding
task acceptance. These core-only ZINC A/B controls do not replace the complete
four-workload campaign against a freshly pinned llama.cpp, nor establish an
all-model performance win.

### Fresh package with the narrower default

The [November 8 UTC package validation](https://zolotukhin.ai/zinc/validation/intel-q8-decode-narrow-default-2026-11-08.json)
tests CI archive `b4204560…ffd02b`, source `4b1d2d73`, tree-identical to
candidate head `7f1556c0`. All 376 base numerical cases, 228 Q8 cases / 114
bitwise pairs and fourteen full-vocabulary checks pass at the unchanged
0.001 tolerance. All seven models pass native/API replay: 431 primary choices
and one previously qualified near-tie branch, with 81 complete HTTP
continuations, both Gemma window sets, context boundaries, cancellation/error
recovery, streaming and Unicode. Logs confirm that only the all-Q8 starter
selects the row kernel; the six larger models retain legacy. Explicit-off and
missing-shader controls retain the native starter output too.

The separate three-round starter run retains all 144 requests, 36 streaming
probes and nine server lifetimes. Short-chat median-of-round-medians decode
rises from **193.20 to 251.22 tokens/s**, about 30%, and HTTP latency falls
from **527.76 to 413.22 ms**. The newly built llama.cpp Vulkan `f477f795`,
selected immediately before its build and checked again at campaign start,
reaches **289.39 decode** and **2,722.97 prefill tokens/s**. Candidate prefill
is 1,925.46 tokens/s. Individual round medians and every sample remain in the
report; this is not an all-model performance win.

Both engines use the same GGUF, F16 KV, MTP off, one slot and uncached raw
completions. Natural EOS on context-long remains visible and excludes that
pair from full-budget comparisons. Startup retains the historical 500 ms
polling protocol and cannot resolve smaller differences. Hardware, package and
reference identities stay unchanged through the run. These are development
archive checks, separate from final release certification and coding-task
acceptance.

## Work order after 0.1.0

1. Keep all seven qualified models selectable through the managed catalog.
   Qwen 3.8's missing Intel profile rejected a qualified model before loading;
   the follow-up catalog fix and CPU regression align selection with the
   existing package qualification. This does not change the published 0.1.0
   archives.
2. Rebuild the freshly pinned upstream llama.cpp Vulkan server and tokenizer on
   the Intel worker. Keep its performance pin separate from the frozen native
   correctness oracle. Record the active kernel, firmware and Mesa stack before
   the campaign and retain that stack through each controlled comparison.
3. Continue tuning the starter after removing its serial fallback, then correct
   Gemma 26B's conservative path and Qwen 35B's Intel prefill. Existing
   experimental batched paths have failed complete native continuations; an
   optimization must fix those failures before becoming a default. Reducing
   expert counts, enabling approximate activations or changing KV precision is
   not a matched performance improvement.
   For the Q8 starter, run `tools/validate_q8_prefill.py` on the remote Intel
   worker with `--zinc-tree`, `--worker-config` and `--output`. It compares the
   complete vocabulary's batched logits with serial F32-activation/F32-KV prefill at
   fourteen token/tile boundaries, including 1, 31/32/33, 95/96/97 and
   383/384/385 tokens.
   A silent fallback, missing comparison, nonfinite result or difference above
   the existing 0.001 tolerance fails. A precision control at two tokens measured
   0.003924 with F16 KV and 0.000033 with F32 KV: cache rounding magnifies small
   summation-order differences. The numerical fixture keeps F32 KV and its
   original tolerance; native continuations, chunk boundaries and API recovery
   must separately pass with the F16 KV used for performance. This numerical
   regression is not a throughput claim. A failed run retains completed cases,
   the active token boundary and its failure reason; unexpected process errors
   retain their type without publishing private worker paths or commands.
4. Profile decode across the dense, hybrid and MoE models, then tune the hot
   matrix-vector, attention and submission paths. Validate shader math on the
   actual Intel GPU, complete native continuations, API recovery and both
   Gemma sliding windows after each change. Keep comments and regression tests
   for the defect or invariant being fixed.
5. Repeat the full matched campaign and actual-client coding checks. Retain
   failures and natural EOS. Resolve the starter's incomplete long-context
   coverage with a workload that both engines complete at the same output
   budget, retaining the old diagnostic. A different cap for each engine or
   silently suppressing EOS cannot establish a matched win.
   The starter's known coding limitations remain
   separate from engine correctness and throughput; a successful process exit
   cannot substitute for independent task acceptance tests.

## Native Intel compute after the Vulkan milestone

Native stack bring-up can proceed while the remaining Vulkan performance gaps
are investigated. The administrator tool at
[`scripts/upgrade_intel_stack.py`](https://github.com/zolotukhin/zinc/blob/main/scripts/upgrade_intel_stack.py)
targets the existing Arch/GRUB worker. Run it without arguments for a plan. Its
explicit `--apply` mode requires root, preserves the running kernel and modules
behind a separate GRUB fallback, saves signed rollback package archives, and
performs a full Arch update without rebooting. System library rollback still
requires reviewing the complete transaction.

Recovery retains the original package-manager failure and reports secondary
module/GRUB errors separately. A same-release successful reinstall keeps its
installed modules; a different kernel image reusing that release stops recovery
for administrator review. Old unowned module directories still receive their
saved contents. Cached and downloaded rollback archives both require valid
signatures, and package verification retains Arch epochs and release suffixes.

The November 7 live plan found Arch kernel `7.2.14` versus upstream stable
`7.3.3`, and Arch oneAPI `2026.0` versus Intel `2026.2`. The script refreshes
the kernel and Arch metadata at runtime; its Intel compiler reference is dated
in the source. Applying a lagging distribution stack requires the explicit
`--allow-distro-lag` option and does not qualify it as the latest upstream stack.
It never adds Ubuntu repositories to Arch or installs a release-candidate kernel.

For repeatable low-level validation, compile `tools/intel_sycl_probe.cpp` with
`icpx -std=c++17 -fsycl -O2 tools/intel_sycl_probe.cpp -o zinc-sycl-probe` on the remote worker after loading
the oneAPI environment. Pass its absolute binary path to
`tools/intel_native_probe.py --probe <binary> --reservation <worker-reservation>
--output <report.json>`, reusing the worker's existing reservation path. The
probe requires exactly one Intel Level Zero GPU and checks both queue modes,
explicit event dependencies, tail elements, copies and output guards. Its
78 cases are readiness evidence; they do not qualify a ZINC inference backend
or establish model throughput.

After the coverage and Vulkan performance work, add a separate native Intel
backend using Intel's Level Zero runtime and compiler toolchain, with SYCL/XMX
kernels where measurements justify them. Intel's
[Open Middleware Xe installation guide](https://dgpu-docs.intel.com/installation-guides/installing-omix.html)
describes the validated compute stack: Level Zero/OpenCL drivers, the SYCL
compiler and compute libraries. Upstream llama.cpp's
[SYCL backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
provides the native Intel competitor for that stage. Native qualification must
compare against that backend as well as retaining the Vulkan control.

First inspect the live system's OS, kernel, Xe driver, firmware, Level Zero
loader/runtime, graphics compiler and oneAPI packages. The last recorded worker
kernel was `7.2.7-arch1-1`, with Mesa `26.2.3-arch1.2`; those are historical
observations, not a live inventory. A newer kernel alone does not establish
native compute readiness. Intel OMIX currently documents Ubuntu 26.04.1,
Ubuntu 24.04.5 with HWE, and RHEL 9.8/10.2. Select a compatible package set for
the actual OS before choosing a kernel or driver change. Do not apply Ubuntu
package instructions to an unverified distribution.

If a kernel or driver upgrade is needed, retain the working kernel boot entry,
the previous package versions and the tested Vulkan executable. Change one
stack component at a time, reboot when required, verify SSH/GPU access and
rerun the same numerical, model and recovery gates. Record the old and new
stack identities with their separate benchmark results. A package-manager
success or `sycl-ls` listing is bring-up evidence, not model qualification.

The native backend is not implemented or qualified by this plan. Its first
gates are device discovery, allocation/copy and command submission, followed
by exact numerical kernels, one end-to-end model, all seven native model
continuations and API/OpenCode recovery. Keep Vulkan and ROCm release builds
independent of the new runtime, and qualify native binaries separately before
advertising native Intel support.

### Official package environment and native reference

The [November 9 seven-model support record](/zinc/validation/intel-seven-model-support-2026-11-09.json)
checks development Vulkan package `56bec228` on the updated Intel node. All seven
catalog models pass: 376 shared numerical cases, 228 Q8 cases with 114 identical
reduction pairs, tokenizer parity, 27 full native-reference continuations, 81
full HTTP replays and 27 raw JSON/SSE checks. Native choices are 431/432 exact
with one qualified Qwen 35B near tie. Both Gemmas cross their 1024-token window;
cancellation and error recovery preserve complete native answers. The binary,
all 214 packaged shaders, kernel and driver remain unchanged through cleanup.
This is a development inference/protocol qualification. Actual OpenCode and
OpenClaw task acceptance and final release-byte certification remain separate.

The same record retains the failed native SYCL FP16 attempt at `a4d4f2fd`,
selected as upstream HEAD when that campaign started. Six models fail repeated
complete-output checks; Qwen 35B exceeds the allowed near-tie count. The frozen
short reference checks pass for the other six, so these findings do not mean
their inference fails outright. A separate FP32 starter diagnostic also drifts
on one medium-context repetition across 24 full requests and eight SSE probes.
Changing FP16 to FP32 alone does not resolve this consistency check. The native
baseline remains unqualified while oneDNN/attention controls are isolated;
these failed attempts establish no headline speed claim. The benchmark pin
has since advanced to `5f490d7b`; rebuilding it is separate from these frozen
results.

The [refreshed native runtime controls](/zinc/validation/intel-native-runtime-controls-2026-11-09.json)
use `5f490d7b` with the same starter GGUF and official Intel environment. Six
treatments cover FP32/FP16 with default oneDNN, native SYCL attention, or oneDNN
disabled. Each gets 24 full requests and eight raw SSE controls across four
workloads and two server lifetimes. FP32 default, FP32 without oneDNN, FP16
with native attention, and FP16 without oneDNN pass this limited repeatability
check. FP32 with native attention drifts twice; default FP16 drifts nine times.
All attempts remain in the record. These starter controls identify candidates
for the full catalog; they do not establish that nondeterminism was fixed or
qualify an all-model native baseline.

On November 9, the remote Arch worker was running kernel `7.2.14-arch1-1`,
compute runtime `26.40.40196.8`, IGC `2.42.6`, Level Zero loader `1.34.0`,
firmware `20261019` and Mesa `26.2.6`. A separate, user-owned environment now
contains Intel's official DPC++ compiler, SYCL runtime and oneMKL `2026.2.0`,
oneDNN package `2026.1.0`, and oneDPL `2022.14.0`. These packages do not replace
the system's oneAPI environment. This is project testing of official packages
on Arch; Arch is not one of Intel OMIX's documented validated distributions.

The probe compiled with that compiler passed all 78 cases, checking 13,406,292
elements and 9,984 output guards with zero numerical error. A diagnostic native
llama.cpp starter-model run fully offloaded 29/29 layers and returned identical
complete output across four repetitions. That establishes bring-up, not a
seven-model performance or coding qualification.

The [explicit Linux x86-64 package lock](https://github.com/zolotukhin/zinc/blob/main/config/intel/oneapi-2026.2-linux-64.explicit.txt)
retains all 41 archive URLs and SHA256 checksums, without installation paths.
After verifying and installing micromamba, create the isolated environment on
the remote worker:

```bash
micromamba create -y --prefix "$ZINC_INTEL_PREFIX" \
  --file config/intel/oneapi-2026.2-linux-64.explicit.txt
```

Intel documents its official package channel in the
[compiler installation instructions](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html).
Keep the compiler and runtime in the same environment. Build the refreshed
llama.cpp source with the official compiler:

```bash
micromamba run --prefix "$ZINC_INTEL_PREFIX" cmake \
  -S "$LLAMA_SOURCE" -B "$LLAMA_SYCL_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_PREFIX_PATH="$ZINC_INTEL_PREFIX" \
  -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_HOST_MEM_FALLBACK=OFF \
  -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TESTS=ON
micromamba run --prefix "$ZINC_INTEL_PREFIX" cmake \
  --build "$LLAMA_SYCL_BUILD" --parallel 2 \
  --target llama-server llama-cli llama-tokenize llama-eval-callback test-backend-ops
```

Also build `GGML_SYCL_F16=OFF` in a different build directory. Upstream recommends
testing both precisions because their speed and accuracy differ. Graph execution
is separately opt-in at runtime; compiled graph support does not mean it is
active. Keep the default and `--sycl-graphs` treatments separate. Do not enable
persistent SYCL caching as an assumed speed fix: upstream documents crashes
with it. See the [upstream SYCL guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md).

For consistency diagnostics, `--sycl-native-attention` selects the native SYCL
attention kernel instead of oneDNN SDPA; `--sycl-disable-dnn` separately selects
the oneDNN-disabled control. Each campaign records all three explicit runtime
settings. The default keeps oneDNN and its attention path enabled and graphs
disabled. Compare these treatments separately with their original failures
retained; a successful slower control does not replace the recorded default.

Run the native comparison on that worker with its existing private reservation
and model configuration:

```bash
python3 tools/benchmark_intel.py \
  --zinc-tree "$ZINC_PACKAGE_TREE" --zinc-source "$ZINC_SOURCE" \
  --llama-bin "$LLAMA_SYCL_BUILD/bin" --llama-source "$LLAMA_SOURCE" \
  --reference-backend sycl --reference-device "$LLAMA_SYCL_DEVICE" \
  --sycl-prefix "$ZINC_INTEL_PREFIX" --sycl-probe "$ZINC_INTEL_PROBE" \
  --oracle "$ZINC_REFERENCE_BIN" --oracle-source "$ZINC_REFERENCE_SOURCE" \
  --worker-config "$ZINC_WORKER_CONFIG" --output "$ZINC_BENCHMARK_REPORT"
```

The harness requires one physical Intel render device, full SYCL offload, the
78-case readiness probe, installed package identities and original CMake build
settings. It verifies the live server's SYCL and Level Zero libraries come from
the selected environment. Those identities are checked again after cleanup.
ZINC and the frozen Vulkan oracle keep their own loader environments.

Conda's OpenCL loader initially exposed only the bundled CPU ICD. Short SYCL
continuations passed, but a 167-token prompt aborted when oneDNN needed its
matching GPU OpenCL device. The reference environment now scopes
`OCL_ICD_VENDORS=/etc/OpenCL/vendors` to the system's existing GPU driver.
This changes no system files. A preflight child requires exactly one Intel
OpenCL GPU and checks its name against Level Zero; CPU-only discovery fails.
Only names, versions and library hashes are retained, never device UUIDs or PCI
identifiers. The fixed FP16 starter diagnostic passes all four workloads:
12 JSON requests and four raw SSE probes, with the Intel GPU OpenCL library
confirmed in the live process. The interrupted campaign remains separate.
The performance server also uses `--cache-ram 0`, so unused automatic RAM prompt
caches cannot inflate the competitor's memory or request latency.

The exact timed SYCL configuration must pass complete frozen-reference
continuations, including Gemma's extended window prompts. An allowed near tie
requires an independently teacher-forced oracle suffix and matching decoded
text; it cannot excuse unrelated later errors. Failed validation blocks speed
claims. All packaged SPIR-V hashes and repeated full JSON/SSE output hashes
are retained. ZINC's backend remains Vulkan in this comparison; this does not
implement a native ZINC Intel backend.
