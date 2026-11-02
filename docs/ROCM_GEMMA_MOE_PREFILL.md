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

ROCm Gemma MoE therefore uses serial prefill by default through the shared
CLI/single-slot HTTP entrypoint. It selects that path before allocating batched
scratch or executing a layer. Prompt positions remain absolute; only the final
prompt token computes the next-token logits. Dense Gemma, Muse, Qwen and CUDA
retain their existing routing.

`ZINC_ROCM_GEMMA_MOE_BATCHED_PREFILL=1` explicitly enables the unqualified batched
path and logs a warning. It is outside the release correctness claim. More
batched-stage isolation is needed before enabling it by default again; the
serial control does not identify one particular batched kernel as the cause.

CPU policy tests guard the model boundary. Qualification still requires the
three complete short continuations and three continuations beyond Gemma's
1,024-token window, through CLI and HTTP, followed by request recovery. No
reference tolerances, prompts or agreement limits were weakened.

## Fresh R9700 measurements

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
phase time in all four. These measurements replace the unqualified September
rates on the dashboard and README chart. The complete September record and its
correctness caution remain in `measurement_history` in the downloadable data.
The batched opt-in remains unqualified.

## Revalidate a published measurement

`tools/benchmark_gemma_rocm.py` runs the correctness gate before collecting
performance samples. It checks twelve tokenizer cases, six complete CLI
continuations and eighteen complete HTTP continuations across initial requests,
cancellation recovery and context/error recovery. Three prompts exceed the
1,024-token sliding window. The numerical oracle uses Vulkan with serial
prefill, F32 K/V and flash attention off; the performance baseline uses native
HIP with batched prefill and flash attention on. Both reference builds must
report the exact current performance-reference commit and their library hashes.
The release reference pin remains separate.

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
  --llama-source "$LLAMA_SOURCE" --model "$GEMMA_GGUF" \
  --device "$ROCR_DEVICE" --oracle-device "$VULKAN_DEVICE" \
  --reservation "$GPU_RESERVATION" --output gemma-rocm.json
```

The reservation must be shared with other jobs using that physical GPU. The
runner refuses a busy worker and stops every process group it starts, including
on failure or cancellation. Its portable report contains hashes and public
prompts; host paths and endpoints are excluded. This model-specific check does
not qualify a stable release or validate the experimental batched path.

`bun tools/merge_gemma_rocm.mjs <report.json> <full-harness-commit>` verifies the
frozen harness sources, all 96 exact CLI choices, recovery checks, payloads and
complete repeated sample matrix before replacing the dashboard row. It archives
the original measurements and their caution together. A short passing output
or a changed warning string cannot clear a historical correctness finding.
