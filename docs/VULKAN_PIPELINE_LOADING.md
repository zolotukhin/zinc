# Vulkan pipeline loading on Intel

ZINC previously attempted all 72 fixed-input-width projection variants when a
model loaded. Most belonged to other matrix shapes. The Vulkan engine now
selects those variants from the input widths of the loaded GGUF matrices.
Generic pipelines remain available, unlisted future widths load conservatively,
and standalone kernel benchmarks retain the complete set.

The host-only selection tests distinguish matrix inputs from output rows,
expert counts and norm vectors. They run on every backend. A separate Linux
contract checks that skipping a variant opens neither its shader file nor a GPU.
Shader code and dispatch workgroup sizes are unchanged.

## Measured effect

The [recorded Intel A/B evidence](/zinc/validation/intel-vulkan-pipeline-loading-2026-11-03.json)
compares main archive `a232b046` with pipeline archive `2b79023c`. Both contain
the same 213 SPIR-V files byte-for-byte. The worker is Intel BMG G31 with Mesa
26.2.3; both servers use the same pinned GGUF, F16 KV, 4K context, one slot,
disabled MTP and disabled prompt caching.

| Observation                                      | Main     | Selected variants |
| ------------------------------------------------ | -------- | ----------------- |
| Qwen 3.5 9B startup, disk shader cache disabled  | 10.326 s | 8.060 s           |
| Qwen 3.5 9B startup, ordinary warm runs          | 1.637 s  | 1.642 s           |
| Gemma 26B startup, ordinary warm runs            | 4.410 s  | 4.383 s           |
| Resident GPU memory saved on each measured model | —        | 4 MiB             |

The first row is a separate startup experiment: three balanced trials per
variant, 20 ms health polling, and a pre-read GGUF before every process. It uses
the [documented Mesa disk-cache control](https://docs.mesa3d.org/envvars.html),
`MESA_SHADER_CACHE_DISABLE=true`. Startup falls about 22% in that scenario.
Ordinary warm startup is effectively unchanged. These measurements do not
establish cold-disk startup or a speed advantage over llama.cpp.

The ordinary A/B uses all four canonical workloads on Qwen 3.5 9B and Gemma
26B: three balanced rounds, one warmup and three measured requests per workload.
All 192 complete responses are identical across variants, including warmups;
48 separate streaming probes also pass. Median decode rates differ by less than
0.3%, while the short Qwen prefill median differs by about 1.8%. The raw timings
remain in the report. The measured benefit is reduced initialization cost.

The large host-visible memory peak is a separate upload problem. Qwen 9B peaks
near 798 MiB during loading but uses 18 MiB once ready; skipping variants does
not reduce that upload peak. Candidate certification and wider hardware claims
still require their own packaged-byte gates.

The same pipeline archive passes 376 numerical cases and all seven models'
tokenizer, complete CLI and initial HTTP reference checks. Both Gemmas include
prefixes beyond their 1,024-token window. Of 432 native choices, 431 match
exactly and one meets the unchanged near-tie policy. These checks use the pinned
release oracle `5f436ddd`; the separate performance reference stays `2cbca389`.
They do not substitute for final-candidate cancellation and recovery gates.
