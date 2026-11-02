# Gemma's Vulkan sliding-window correction

Gemma 4's sliding layers must discard keys at least 1,024 positions behind the
current query. Its global layers retain the complete causal prefix. The pinned
reference uses the [STANDARD mask](https://github.com/ggml-org/llama.cpp/blob/5f436dddb440a288ee5611d7d1eca564a6aca9f4/src/llama-hparams.h)
and [marks Gemma 4's layers explicitly](https://github.com/ggml-org/llama.cpp/blob/5f436dddb440a288ee5611d7d1eca564a6aca9f4/src/models/gemma4.cpp).

Vulkan attention previously read the whole prefix for both layer types. Short
reference continuations passed because they never crossed the sliding window.
Actual Intel OpenCode sessions with approximately 6K-token prompts exposed
repetitive output and no tool calls, even after the cache-allocation fix let
the models load at larger contexts. A successful client exit with unchanged,
failing acceptance tests remains a failed coding task.

## Correct bounds for every query

The window includes the query token. With a 1,024-token window, causal lengths
1,023 and 1,024 start at key zero; length 1,025 starts at key one. A prefill
batch crossing that boundary needs a different lower bound for each query.
The bound applies to logical positions before looking up physical KV pages.

The Vulkan push constants now carry `sliding_window_size`. Regular attention,
split attention and F32/F16 batched attention honor it. A zero window preserves
full history. Gemma 4's smaller-head sliding layers receive its model window;
the global layers receive zero, matching the existing per-layer RoPE choice.
Optimized readers that do not implement the mask remain available for global
attention. Sliding layers use the regular readers; optional Q8 caching uses
serial prefill and the window-aware Q8 decode shader.

Dense Gemma also keeps F32 activations during batched prefill by default.
A control on the extended copper prompt found an unqualified continuation
change with the extra Q8 activation approximation. Disabling that prefill
approximation restored the exact continuation; disabling decode quantization
alone did not. `ZINC_GEMMA_PREFILL_DP4A=1` explicitly opts into approximate
prefill, with a diagnostic warning. Weight quantization and F16 KV caching
are unchanged.

## Tests that cross the boundary

The numerical GPU validator compares every output element with an independent
CPU softmax oracle. Its **256 cases** cover 256/512-element heads, causal lengths
1,023/1,024/1,025/1,041, windows zero/one/17/1,024, permuted 16-token pages,
F32/F16/Q8 cache storage, regular and split attention, batched queries, output
guards and replay. The old shipped shaders fail the negative control on the
R9700; the corrected shaders pass all 256 cases on that board.

Both Vulkan release roles must run those cases in addition to the existing
84 cache-boundary and 36 Gemma-router cases: **376 numerical cases** in total.

Model qualification also includes three frozen reference continuations with
prompts beyond each Gemma's policy window, through both CLI and HTTP. Reports
retain measured prompt-token bounds and the existing reference-agreement
policy. Aggregation rejects missing, failed or shorter-than-window evidence.
These checks supplement the short continuations and request-recovery checks.

## Repeating the benchmark after a correctness fix

The two Gemma models use `tools/benchmark_gemma_rocm.py --backend vulkan` for
remeasurement on AMD and Intel. The runner shares the canonical workload
payloads, parsers and statistics with `tools/performance_suite.mjs` and the
ROCm revalidation. Vulkan uses its normal F16 K/V cache in both engines. Dense
Gemma keeps the corrected F32 activation prefill default; the experimental
DP4A override is not set.

Each model must first pass 376 numerical shader cases, twelve tokenizer cases,
six complete CLI reference continuations and eighteen complete HTTP reference
continuations. Three prompts cross the 1,024-token window. HTTP checks repeat
the full continuations after cancellation and after context/error recovery.
The numerical oracle uses native Vulkan, serial prefill, F32 K/V and flash
attention off, with the existing reference agreement policy unchanged.
Its source, executable and libraries are recorded separately from the timing
baseline. The oracle may use the campaign revision or the fixed release
correctness pin in `config/release/targets.json`; arbitrary older references
are rejected. Throughput always uses the current campaign revision in
`config/benchmark-reference.json`.

Performance then uses the normal batched reference with flash attention on.
Both engines serve the same pinned GGUF and full prompts with a 4K context,
one slot, MTP off and prompt caching disabled. Three rounds alternate engine
order, with one discarded warmup and three measured requests per workload
and round: nine measured requests per engine and workload, 96 requests total
per model. Repeated full responses must be identical within each engine.

Select the latest llama.cpp revision before building the campaign, then freeze
its executable and libraries across both models. The report records both the
selection time and the upstream HEAD observed at each model's start. A later
upstream change does not change the baseline halfway through a campaign.
`--reference-selected-at` must match the checked timestamp in the committed
pin and be at most two hours old. Without that option, the runner requires
the pin to match live upstream HEAD at startup.

Use a packaged ZINC build or embed its exact source commit with `-Dcommit`
when building from source; a binary reporting `commit: unknown` is rejected.
Supply the observed Vulkan index, package/reference paths, pinned model and
shared GPU reservation through environment variables on the remote worker:

```bash
python3 tools/benchmark_gemma_rocm.py \
  --backend vulkan --target rdna --model-id gemma4-26b-a4b-q4k-m \
  --zinc-tree "$ZINC_PACKAGE" --zinc-source "$ZINC_SOURCE" \
  --llama-bin "$LLAMA_VULKAN_BIN" --oracle "$ORACLE_VULKAN_BIN" \
  --oracle-source "$ORACLE_SOURCE" \
  --llama-source "$LLAMA_SOURCE" --model "$GEMMA_GGUF" \
  --device "$VULKAN_INDEX" --oracle-device "Vulkan$VULKAN_INDEX" \
  --reservation "$GPU_RESERVATION" --output gemma-vulkan.json
```

Use `--model-id gemma4-31b-q4k-m` with its matching model file for dense Gemma,
or `--target intel` for the Intel worker. Each process is supervised and reaped
before its owning command returns. The publisher checks the complete report
and frozen harness before replacing one model's results; the original record
and its caution move together into `measurement_history`.

The [Intel OpenCode investigation](INTEL_OPENCODE_VALIDATION.md) records actual
coding outcomes and their source-specific scope. Numerical agreement alone
does not establish that a model completes a coding task.

### A moving reference can change the continuation

The November 2 dense-Gemma check on the R9700 exposed a difference between the
two reference revisions. With serial prefill and F32 K/V, llama.cpp `1b6935b227b9` chose
` Wiring` at generated token 12 of the extended copper prompt. The fixed
release reference `5f436dddb440` and ZINC's default Vulkan path chose ` wiring`
and then matched through the complete 16-token continuation. The newer
reference's log-probability gap was 0.13855, exceeding the existing 0.05
near-tie limit. The first attempt correctly stopped before collecting timings.

The [complete observation and both reference identities](/zinc/benchmarks/2026-11-02-gemma31-reference-observation.json)
retain the prompt, full token sequences and top-two probabilities. Revalidation
uses the fixed release oracle for this model and keeps `1b6935b227b9` for the
performance comparison. No agreement threshold changes, alternate prompts or
approximate prefill overrides are needed. This observation establishes a
source-dependent continuation difference, not which capitalization is the
better answer.

## Fresh Vulkan measurements

On November 2, 2026, both Gemmas completed the revalidation protocol on the
R9700 and Intel Arc / BMG G31. Every pair passed 376 numerical cases, twelve
tokenizer cases, all 96 CLI token choices exactly and eighteen complete HTTP
continuations with recovery. The four performance campaigns completed **384
requests**: 288 measured requests and 96 discarded warmups.

Both engines use F16 K/V, a 4K context, one slot, MTP off, prompt caching off
and the same checksum-pinned GGUF for each pair. ZINC uses its shipped Vulkan
prefill defaults, including serial MoE prefill on Intel and F32 activation
prefill for dense Gemma. No approximate prefill overrides are enabled.

The table shows median quick-chat rates as **ZINC / llama.cpp** in tok/s.
Overall compares combined prefill and decode phase time; above 1.00× favors
ZINC. Incident context uses the longer, 353-token workload.

| GPU                 | Gemma       | Prefill        | Decode        | Overall, quick chat | Overall, incident context |
| ------------------- | ----------- | -------------- | ------------- | ------------------- | ------------------------- |
| R9700               | 26B-A4B MoE | 1435.8 / 444.5 | 134.4 / 104.9 | 1.39×               | 1.26×                     |
| R9700               | 31B dense   | 138.3 / 167.0  | 30.7 / 29.2   | 1.03×               | 0.88×                     |
| Intel Arc / BMG G31 | 26B-A4B MoE | 66.7 / 217.7   | 63.9 / 81.4   | 0.61×               | 0.28×                     |
| Intel Arc / BMG G31 | 31B dense   | 52.7 / 47.8    | 19.6 / 21.9   | 0.93×               | 0.60×                     |

Faster decode does not make every workload faster. On the R9700, dense Gemma's
corrected prefill trails the reference, and its incident-context request takes
longer overall. Intel's conservative MoE prefill also carries a substantial
cost. These rates replace the affected September records. Their original
measurements and cautions remain in `measurement_history`; inherited old
dates and sources are made explicit so a newer platform refresh cannot
relabel them.

ZINC's native source is `0e1f84bbb4aa`. Every timing baseline is llama.cpp
`1b6935b227b9`, verified at 16:16 UTC before the campaign builds and frozen
through all four comparisons. Each evidence file records full source commits,
GGUF checksums, executable and shared-library hashes, driver/OS metadata,
individual requests and its separate correctness reference:

- [AMD R9700 26B-A4B](/zinc/benchmarks/2026-11-02T16-27-35-rdna-gemma4-26b-a4b-q4k-m-revalidated.json): correctness reference `1b6935b227b9`, runner `c5138c6a5d62`.
- [AMD R9700 31B](/zinc/benchmarks/2026-11-02T17-08-32-rdna-gemma4-31b-q4k-m-revalidated.json): correctness reference `5f436dddb440`, runner `cad5fa498427`.
- [Intel Arc 26B-A4B](/zinc/benchmarks/2026-11-02T16-59-20-intel-gemma4-26b-a4b-q4k-m-revalidated.json): correctness reference `1b6935b227b9`, runner `c5138c6a5d62`.
- [Intel Arc 31B](/zinc/benchmarks/2026-11-02T17-31-56-intel-gemma4-31b-q4k-m-revalidated.json): correctness reference `1b6935b227b9`, runner `c5138c6a5d62`.

## Worker readiness follow-up

The preview worker initially copied model identity and context while dropping
the pinned window field. Its extended-check branch therefore never ran, even
though candidate aggregation already required that evidence. Readiness now
preserves the boundary. A CPU lifecycle regression exercises the real
pin-to-execution wiring and rejects a long-window disagreement after otherwise
passing short checks; restoring the old wiring fails both controls.

The [fresh automatic R9700 reports](/zinc/validation/worker-readiness-window-gate-2026-11-02.json)
include both Gemmas' measured 1,136–1,139-token prompts, CLI agreement and full
HTTP continuations. Vulkan matches all 432 short/extended choices across seven
models. ROCm matches 382 of 384 exactly, with two policy-qualified near-ties.
These preview reports retain their archive and validation-source identities;
they do not certify a final candidate or an Intel worker.
