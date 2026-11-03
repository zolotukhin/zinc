# ROCm Gemma MoE prefill correctness

The extended reference gate exposed a Gemma 4 26B-A4B mismatch in the ROCm
batched prefill path. All 48 short reference choices matched. After a 1,136-token
prefix, the copper continuation began with `Here` instead of `Three`. The
reference log-probability gap was about 2.274, beyond the unchanged 0.05
near-tie allowance. This result cannot qualify a release archive.

Controls retained the same packaged executable, weights, prompt and native
reference. Disabling only Q8 projection prefill or F16 attention did not fix
the continuation. Full serial prefill matched all 16 choices, including when
prompt-internal tokens skipped the unused output head.

The November 3 investigation identified concrete faults in the accelerated
path. The Q5_1 grouped WMMA kernels counted 128-column tiles with integer
truncation, although the weight format requires only 32-column alignment.
Gemma's 704-column expert down projection therefore lost its final 64 input
columns. Q8_0 WMMA had the same defect for inputs such as the 2,112-column
shared expert. Both now process the final partial tile and mask missing groups.

The packed activation quantizer now zeros missing scale/sum headers and skips
spare launch tiles beyond the packed allocation. Batched routing keeps its F32
weights and activations even when the projection pre-dequantization cache is
active. The serial-expert diagnostic also computes its own normalization;
batched-attention scratch cannot substitute for the scalar scratch buffers.

Batched Gemma MoE prefill is restored as the default. Set
`ZINC_ROCM_GEMMA_MOE_BATCHED_PREFILL=0` for the retained serial control. The
fixed source matches all 96 native CLI choices over three short prompts and
three prompts beyond the 1,024-token window, plus 18 complete HTTP
continuations before and after cancellation and context/error recovery.
Reference prompts, tolerances and agreement limits are unchanged.

`tools/validate_rocm_partial_tiles.c` adds 305 GPU cases against llama.cpp's
native Q5_1/Q8_0 dequantizers. Tail-only weights make the old kernels return zero
where the reference is nonzero; the fixed kernels pass. The
[negative-control records](/zinc/validation/rocm-gemma-partial-tiles-2026-11-03.json)
retain both old failures and the separate prototype fixture hashes. The matrix covers all
seven Q8_0 tile variants, three grouped Q5_1 variants, production widths,
partial output rows and token tiles, nonzero offsets, expert routes, poisoned
padding, output guards and quantizer row folding. CPU fixture checks run in the
ordinary test suite. ROCm release qualification also requires the complete GPU
matrix and verifies that the tested kernel source is embedded in the archive's
executable. Passing source diagnostics alone does not qualify a release archive.

## Fresh R9700 measurements

The [November 3 complete report](/zinc/benchmarks/2026-11-03T09-00-39-rdna-rocm-gemma4-26b-a4b-q4k-m-revalidated.json)
records ZINC and harness `65363680c9e9`, llama.cpp HIP `36ee3caf7aa2` verified
as upstream HEAD before the campaign's build, and the separate Vulkan
correctness oracle `5f436dddb440`. Both performance engines use ROCm, the same
GGUF, F32 K/V, 4K context, one slot, prompt caching off and MTP off.
The embedded-kernel gate passes all 305 cases; all 96 native choices and
18 HTTP recovery continuations match. All 96 benchmark requests complete:
72 measured and 24 discarded warmups, over three alternating engine rounds.

| Workload          | ZINC prefill | llama.cpp prefill | ZINC decode | llama.cpp decode | Overall |
| ----------------- | -----------: | ----------------: | ----------: | ---------------: | ------: |
| Quick chat        |     1,187.28 |            780.56 |      102.14 |            81.40 |   1.27× |
| Coding review     |     2,895.47 |          1,857.46 |      100.16 |            77.08 |   1.31× |
| Incident context  |     3,702.69 |          2,541.52 |       98.77 |            75.14 |   1.32× |
| Long coding draft |     1,486.25 |            986.03 |      100.63 |            79.30 |   1.27× |

Rates are median tokens per second over nine measured requests per cell.
Overall is llama.cpp's combined prefill/decode phase time divided by ZINC's.
ZINC leads prefill, decode and combined phase time in all four workloads.
The incident-context prefill rate rises from 98.04 to 3,702.69 tok/s;
the two campaigns retain their different reference revisions and dates.
These are frozen source-build measurements, not stable-release qualification.
The previous serial and historical unqualified batched measurements remain
archived with their original settings and correctness findings.

## Packaged regression sweep

The [CI archive sweep](/zinc/validation/rocm-gemma-batched-package-sweep-2026-11-03.json)
uses the exact ROCm archive from packaging run
[41606559665](https://github.com/zolotukhin/zinc/actions/runs/41606559665),
source `c09401c93753`, with native HIP correctness reference `5f436dddb440`.
All six release models pass their tokenizer, reference continuation, HTTP and
recovery checks. Across short and sliding-window continuations, **383/384 token
choices are exact**. Qwen 3.6 MoE's remaining choice has a native log-probability
gap of 0.0321, within the unchanged 0.05 allowance. Gemma 31B also passes all
three forced 16-token prefill-chunk continuations. The 305 embedded-source
numerical cases pass, and every owned GPU process is stopped afterward.
This preview readiness report remains separate from final release certification.

## Cache controls and actual OpenCode

The [final-comment archive report](/zinc/validation/rocm-gemma-batched-opencode-2026-11-03.json)
records packaging run
[41607547761](https://github.com/zolotukhin/zinc/actions/runs/41607547761), source
`4db4a70bd8fc`, and archive/binary/model/reference hashes. This archive also
passes all 305 numerical cases, 96 exact default-cache continuation tokens and
18 HTTP recovery continuations. Disabling the projection cache preserves all
96 choices. The serial-expert diagnostic independently preserves the 16-token
extended copper continuation.

At **8K context**, the same archive passes **26/26 API checks** and all three
core coding fixtures using **OpenCode 1.19.4**. Each fixture starts a fresh
single-slot server, with F32 K/V, MTP and prompt caching off, thinking off,
a 2,048-token output cap, 16 steps and a 600-second budget.

| Fixture           | Independent tests | Tool calls | Seconds |
| ----------------- | ----------------- | ---------: | ------: |
| Rate limiter      | 3/3               |         11 |  63.974 |
| Multi-file cart   | 3/3               |         10 |  44.882 |
| Python NDJSON CLI | 4/4               |          6 |  31.604 |

All ten independent tests pass. The runs record no API errors, failed tools,
timeouts, malformed commands, protected-file changes, unexpected files,
protocol repairs or suppressed content. Final input prompts contain
4,551–5,194 tokens. These small fixture results retain their own client,
context and budget; final multi-profile release qualification remains separate.

## November 2 serial fallback measurements

The November 2, 2026 revalidation passes **96/96 exact CLI token choices**,
all **18 complete HTTP continuations** and twelve tokenizer cases. Extended
prompts contain 1,136–1,139 tokens. The same continuations still match after
stream cancellation, a near-capacity request and rejected invalid requests.
The subsequent benchmark completes all 96 requests: 72 measured and 24 warmups.

The [complete report](/zinc/benchmarks/2026-11-02T05-33-10-gemma-rocm-revalidated.json)
records the pinned GGUF, native ZINC `34f33b900969`, frozen harness
`93f431d40d2b`, and llama.cpp `785322d17f2d`, verified as upstream HEAD when
the campaign began. Both performance engines use ROCm. The separate numerical
oracle uses Vulkan at that same llama.cpp revision.

| Workload          | ZINC prefill | llama.cpp prefill | ZINC decode | llama.cpp decode | Overall |
| ----------------- | -----------: | ----------------: | ----------: | ---------------: | ------: |
| Quick chat        |       111.18 |            789.58 |       91.02 |            81.49 |   0.79× |
| Coding review     |       105.11 |          1,859.01 |       79.09 |            77.23 |   0.55× |
| Incident context  |        98.04 |          2,539.48 |       72.43 |            75.29 |   0.34× |
| Long coding draft |       110.27 |            992.45 |       83.55 |            79.45 |   0.87× |

Rates are median tokens per second over nine measured requests per cell.
Overall is llama.cpp's combined prefill/decode phase time divided by ZINC's;
higher favors ZINC. The corrected serial path gives up substantial prefill
speed. ZINC decodes faster in three workloads, while llama.cpp wins combined
phase time in all four. These measurements originally replaced the unqualified September
rates on the dashboard and README chart. The complete September record and its
correctness caution remain in `measurement_history` in the downloadable data.
The November 3 batched campaign supersedes these serial rates; the full serial
record remains in measurement history.

## Revalidate a published measurement

`tools/benchmark_gemma_rocm.py` runs the correctness gate before collecting
performance samples. It checks twelve tokenizer cases, six complete CLI
continuations and eighteen complete HTTP continuations across initial requests,
cancellation recovery and context/error recovery. Three prompts exceed the
1,024-token sliding window. The numerical oracle uses Vulkan with serial
prefill, F32 K/V and flash attention off; the performance baseline uses native
HIP with batched prefill and flash attention on. Both reference builds report their exact commits and library hashes. The
correctness build may use the separate frozen release pin; performance uses
the current pre-campaign upstream revision.

After those checks, the runner alternates ZINC and llama.cpp server order for
three rounds of all four canonical workloads. Each server stays loaded across
its workloads. One warmup per workload and round is discarded; nine measured
requests per engine and workload remain. Both use the same pinned GGUF, full
prompt, output budget, 4K context, F32 K/V, one slot and MTP off. Prompt caching
is disabled. Full responses must repeat deterministically within each engine.

Build the CPU tokenizer tool from the same source as the ZINC executable if the
package does not include it. Run this command **on the remote GPU worker**, with
the paths and observed device indices supplied by your environment:

```bash
python3 tools/benchmark_gemma_rocm.py \
  --zinc-tree "$ZINC_PACKAGE" --zinc-source "$ZINC_SOURCE" \
  --llama-bin "$LLAMA_HIP_BIN" --oracle "$LLAMA_VULKAN_BIN" \
  --llama-source "$LLAMA_SOURCE" --oracle-source "$LLAMA_VULKAN_SOURCE" \
  --model "$GEMMA_GGUF" \
  --device "$ROCR_DEVICE" --oracle-device "$VULKAN_DEVICE" \
  --reservation "$GPU_RESERVATION" --output gemma-rocm.json
```

The reservation must be shared with other jobs using that physical GPU. The
runner refuses a busy worker and stops every process group it starts, including
on failure or cancellation. Its portable report contains hashes and public
prompts; host paths and endpoints are excluded. This model-specific check does
not qualify a stable release. New batched campaigns also run the 305 embedded
ROCm numerical checks before collecting performance samples.

`bun tools/merge_gemma_rocm.mjs <report.json> <full-harness-commit>` verifies the
frozen harness sources, all 96 exact CLI choices, recovery checks, payloads and
complete repeated sample matrix before replacing the dashboard row. It archives
the original measurements and their caution together. A short passing output
or a changed warning string cannot clear a historical correctness finding.
