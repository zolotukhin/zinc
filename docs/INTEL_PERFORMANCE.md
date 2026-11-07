# Intel Arc performance qualification

The target is ZINC Vulkan against llama.cpp Vulkan on the same Intel Arc GPU.
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
every 500 ms, so sub-500 ms startup differences are unresolved. Linux process status
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
500 ms polling resolution are unresolved. Ties are not wins.
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
