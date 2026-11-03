# Bounded Vulkan model uploads

Vulkan model loading now reuses one staging buffer, capped at 64 MiB. Previously,
each device-local tensor needed a temporary host-visible buffer as large as the
whole tensor. Large embeddings and expert weights made that allocation hundreds
of MiB even though the buffer was released after the copy.

The loader copies exact byte windows into the final device-local allocation and
waits for each transfer before overwriting the staging buffer. The last window
copies only the remaining bytes. Smaller models reserve only their largest
device-local tensor, up to the cap. Host expert offload and appended NextN tensor
selection retain their existing behavior.

The shared host tests reconstruct a tensor with surrounding guards and exercise
empty, exact, partial and near-address-limit windows. A Linux contract rejects
invalid and overflowing Vulkan copy ranges before any GPU call. Allocation,
recording and submission failures return errors. A failed completion wait
terminates the process without unwinding transfer resources: host/device OOM,
device loss and other wait failures do not establish that the copy has finished.
This prevents pending command, staging and destination buffers from being freed.
The loader also reports a missing staging window explicitly if its selection
invariant is ever broken.
The [Vulkan wait contract](https://docs.vulkan.org/refpages/latest/refpages/source/vkQueueWaitIdle.html)
lists these distinct failure results; a wait error is not proof of completion.

The CPU-only `upload-copy-contract` fixture replaces every Vulkan entrypoint
it calls and never opens a GPU. It verifies exact offsets/bytes/guards on success,
cleanup before submission, and process termination without resource cleanup for
four completion-wait failures. Its subprocess assertions catch an unsafe return
that would execute the command buffer or caller's buffer teardown defers.

## Intel measurements

The [three-round A/B record](/zinc/validation/intel-vulkan-upload-staging-2026-11-03.json)
compares main archive `a232b046` with upload archive `0d3c441e` on Intel BMG G31,
Mesa 26.2.3. Both archives contain identical copies of all 213 SPIR-V files.
This is an isolated upload change; the separate
[pipeline-selection optimization](VULKAN_PIPELINE_LOADING.md) is absent from
both archives.

| Model       | Previous observed GTT peak | Bounded upload peak | GTT once ready, both variants |
| ----------- | -------------------------- | ------------------- | ----------------------------- |
| Qwen 3.5 9B | 797.7 MiB                  | 66.0 MiB            | 18.0 MiB                      |
| Gemma 26B   | 754.0 MiB                  | 66.0 MiB            | 20.0 MiB                      |

Each peak repeats in all three process runs. These are host-visible GPU memory
counters sampled from DRM fdinfo every 50 ms, not total VRAM or whole-process
RSS. Sampling can miss a shorter peak. The reduction is about 92% and 91%,
respectively; steady weight and KV allocations are unchanged.

With OS caches retained, the second and third process starts take 1.64–2.33 s
before the change and 0.83–0.85 s after it for Qwen 9B. Gemma takes 4.47–4.49 s
before and about 2.28 s after. Health is polled every 20 ms. The record also
retains the first process starts (16.55/1.08 s and 42.23/2.90 s). Their initial
cache states are uncontrolled, so they do not establish a cold-disk speedup.

Both variants use the same pinned GGUF, F16 KV, 4,096-token context, one slot,
disabled MTP and disabled prompt caching. Each of four canonical workloads gets
one warmup and three measured requests per round, with balanced variant order.
All 192 complete responses, including warmups, match across variants; all 48
separate streaming probes pass. Per-workload median decode changes are within
0.31%; prefill changes are within 1.12%. This change improves upload allocation
and observed loading time; it does not establish an inference-throughput win.

The A/B archive precedes a standalone-tool import fix. That fix keeps Vulkan
copy bounds inside the Vulkan module so standalone validators can compile;
the upload planner, tensor copies and shaders are unchanged.

The later fixed archive, `f91a63f4`, passes 376 numerical cases and all seven
models' tokenizer, complete short/extended CLI and initial HTTP checks. Both
Gemmas cross their 1,024-token sliding window. Of 432 native choices, 431 match
exactly and one meets the unchanged near-tie policy. The evidence retains both
exact archives, reference identities and the operator experiment sources.
Those earlier checks cover initial HTTP continuations.

The [fresh combined-preview sweep](/zinc/validation/intel-vulkan-upload-final-checks-2026-11-03.json)
tests archive `dc24bbbc6f35` from
[packaging run 41664659161](https://github.com/zolotukhin/zinc/actions/runs/41664659161),
including the merged pipeline selection and the review fixes above. All seven
Intel models pass 376 numerical cases, tokenizer parity, complete short and
sliding-window continuations, HTTP, cancellation and context/error recovery.
Of 432 native token choices, 431 match exactly; the remaining Qwen MoE choice
has a 0.0155 log-probability gap within the unchanged 0.05 allowance. The flagship
also passes active-prefill cancellation and recovers its slot in 2,406 ms.
Every owned GPU process is stopped afterward. The downloadable report records
archive, executable, model, reference and validation-source identities.

This is preview readiness on one Intel GPU. Final release certification still
requires one frozen candidate across all mandatory Linux profiles.
