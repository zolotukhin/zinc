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
