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

`all_models_all_dimensions_better` requires the complete seven-model matrix,
all four full-budget workloads, at least three rounds with three measurements
per engine/workload, and improvement in every measured dimension in every round.
The matched single-slot F16 KV, MTP-off and disabled-cache settings must also
hold. The existing `passed` field records correctness and measurement validity;
it can be true while the performance objective remains false. A diagnostic
subset, an early-EOS workload, an unknown memory counter or a faster pooled
median cannot establish the full objective. Repeated observed improvements do
not themselves establish statistical confidence or concurrent-client capacity.

New campaigns probe GPU, kernel and Vulkan driver identities again after all
servers have stopped. A changed identity fails the campaign; `hardware_unchanged`
records the successful end check. Stack upgrades belong between campaigns.

The [pipeline-loading investigation](VULKAN_PIPELINE_LOADING.md) records a
controlled initialization A/B, including its separate driver-cache experiment
and the unchanged inference throughput.

## Recorded diagnostic and next bottlenecks

The [seven-model GPU diagnostic](/zinc/validation/intel-vulkan-performance-diagnostic-2026-11-03.json)
uses ZINC archive `8ad0f016`, performance reference `2cbca389` and correctness
oracle `5f436ddd` on Intel BMG G31 with Mesa 26.2.3. Both inference engines use
Vulkan and fully offload the model to the same GPU. One round with one warmup
and one measured request per workload establishes directions for optimization;
it is not the default three-round headline campaign.

Short-chat rates below are tokens per second. The linked record retains all
four workloads, exact prompts, output counts, source and archive identities,
memory samples, startup and streaming latency.

| Model            | ZINC prefill | llama.cpp prefill | ZINC decode | llama.cpp decode |
| ---------------- | ------------ | ----------------- | ----------- | ---------------- |
| Qwen 3.5 9B      | 395.5        | 182.1             | 64.1        | 68.3             |
| Gemma 26B        | 66.7         | 217.6             | 63.9        | 81.5             |
| Gemma 31B        | 52.7         | 47.7              | 19.6        | 21.9             |
| Qwen 3.6 35B A3B | 99.3         | 190.0             | 87.0        | 89.8             |
| Qwen 3.6 27B     | 144.7        | 45.5              | 21.5        | 24.2             |
| Qwen 3.8 27B     | 148.0        | 45.5              | 21.5        | 24.3             |
| Qwen 3 0.6B      | 246.9        | 2777.2            | 193.2       | 290.9            |

The 376 numerical cases pass, as do all seven models' tokenizer, complete
continuation and API recovery controls. The small starter naturally reaches EOS in the long-context
workload, so that pair is retained but excluded from full-budget comparisons.
The report therefore has a complete model matrix and an incomplete performance
matrix. It does not establish that every model is suitable for coding tasks.

Decode remains behind the native GPU reference on every model. Prefill is mixed:
Gemma 26B's conservative serial path and the small starter are especially large
gaps. Qualifying faster paths requires complete native continuations and recovery
checks; isolated shader math agreement alone is insufficient. After a change
passes those controls, repeat the matched three-round campaign before claiming
a performance win.

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
3. Profile and correct the largest measured prefill gaps: the starter's serial
   work, Gemma 26B's conservative path, and Qwen 35B's Intel prefill. Existing
   experimental batched paths have failed complete native continuations; an
   optimization must fix those failures before becoming a default. Reducing
   expert counts, enabling approximate activations or changing KV precision is
   not a matched performance improvement.
   For the Q8 starter, run `tools/validate_q8_prefill.py` on the remote Intel
   worker with `--zinc-tree`, `--worker-config` and `--output`. It compares the
   complete vocabulary's batched logits with serial F32-activation prefill at
   eleven token/tile boundaries, including 1, 31/32/33 and 95/96/97 tokens.
   A silent fallback, missing comparison, nonfinite result or difference above
   the existing 0.001 tolerance fails. This opt-in regression supplements the
   native continuations and API recovery gates; it is not a throughput claim.
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
