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
three measured requests per round. Incomplete generations, reduced prefill work,
different prompt counts, and unstable repeated outputs fail the campaign.
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
ready, with OS caches retained; it is not a cold-disk claim. Linux process status
and DRM fdinfo supply host RSS and observed GPU resident/allocated memory peaks.
GPU peaks are sampled every 50 ms, and unavailable counters remain unknown.
Client identifiers and PCI addresses are never published. The counter semantics
come from the [Linux DRM usage statistics documentation](https://docs.kernel.org/next/gpu/drm-usage-stats.html).

A subset selected with `--models`, or fewer repetitions, is a diagnostic. Passing
means that correctness and measurement contracts passed; it does not mean ZINC
beat llama.cpp. Review all model/workload results, retained bytes and settings
before publishing comparisons. This single-slot protocol also does not establish
concurrent-client capacity or MTP-on performance; those need separate matched
campaigns. Release candidate qualification remains a separate gate on the final
packaged bytes.
