# AMD HIP / ROCm setup

HIP / ROCm is the preferred backend for AMD RDNA. Use the separate `rocm`
archive and verify that the installed runtime supports your GPU target.
Vulkan remains an alternative. For installation and the
first prompt, follow [Getting Started](GETTING_STARTED.md). For serving and
clients, follow [Run models and connect apps](RUNNING_ZINC.md#start-the-api).

Each executable contains one backend. A ROCm binary uses HIP and native AMD
kernels; it does not use Vulkan shaders, Mesa RADV, `glslc` or `RADV_PERFTEST`.
Check `zinc --version` before applying backend-specific settings.

## Runtime prerequisites

The stable 0.1.1 ROCm archive needs Linux x86_64 with **glibc 2.39+** and the
validated **ROCm 7.2.4** stack: HIP, hipRTC, hipBLAS, HIP headers and rocWMMA
headers. Ubuntu 24.04 is the reference userspace. Even the prebuilt archive
compiles embedded kernels at runtime, so runtime libraries alone are insufficient.

Install packages for your GPU and distribution using AMD's
[ROCm installation guide](https://rocm.docs.amd.com/en/latest/install/rocm.html).
Select the qualified version above; upstream's latest instructions may target
a newer stack. Then verify that the runtime exposes the intended GPU:

```bash
rocminfo
zinc --version
zinc --check
```

The release reference is the Radeon AI PRO R9700 (`gfx1201`) with Linux kernel
7.2.2 and ROCm 7.2.4. That kernel version records the measured machine; it is not
a fixed build requirement. The running `amdgpu` driver, ROCr runtime and ROCm
userspace must support the same GPU target. This evidence does not qualify
every RDNA3/RDNA4 card. See the
[exact release results](/zinc/validation/release-candidate-0.1.1-2026-11-16.json).

## Select a GPU

Set `ROCR_VISIBLE_DEVICES` before launching ZINC to restrict the native device
list. Set `ZINC_GPU` to your own machine's ROCm device index from `rocminfo`,
then verify the selection:

```bash
ROCR_VISIBLE_DEVICES="${ZINC_GPU:?Set the ROCm device index}" zinc --check
```

Use that same visibility setting for subsequent commands. ROCm currently
ignores `-d`; Vulkan's device indices are not HIP indices. The runtime chooses
the best visible native device.

## Serving limits

Use the [shared server recipe](RUNNING_ZINC.md#start-the-api) with an explicit
context and `--parallel 1` initially. ROCm otherwise defaults to **2,048 tokens
per slot**, even with `-c 0`. Multiple slots need separate cache capacity;
Gemma 4 26B-A4B currently uses one slot regardless of a larger requested value.

The stable 0.1.1 server provides CLI/HTTP chat, raw completions, health and loaded-model
discovery. That release has no browser chat or hot model switching. Sampling is
greedy; `temperature`, `top_p` and client `stop` strings do not change it.
Set `stream` explicitly because ROCm defaults to SSE. The
[API matrix](API.md#backend-coverage) is the canonical endpoint reference.

## Development: browser chat on HIP

Current source builds include `/` and `/chat` on the native HIP server.
After a [ROCm source build](#build-from-source), start the loaded model with:

```bash
zinc chat --model-id qwen38-27b-q4k-m -c 65536 --parallel 1
```

This interface chats with the startup model. Restart the server to change it;
hot model switching and sampling controls remain unavailable on HIP.
Use `--no-browser` over SSH and forward the server port to your computer.
This feature is not in the published 0.1.1 archive. A remote preview may serve
an updated UI separately while using the same native HIP API.

The [R9700 browser/API preview check](/zinc/validation/rocm-browser-chat-2026-11-17.json)
records 64K F32 capacity, a 21,040-token synthetic retrieval and a fresh request
afterwards. It is development evidence, separate from stable release qualification.

## Context, KV cache and MTP

Start with default precision and the release's tested model/context rows.
A successful small prompt does not qualify a long coding session.

The 16K examples are conservative starting configurations, not a model limit.
On the 32 GB R9700, the separate Qwen 3.8 27B investigation records a 64K F32
KV run and a 262,144-token Q8 run. F32 KV at 64K uses about 8.5 GiB including
the NextN attention block, in addition to weights and runtime memory. The full
native window needs opt-in Q8 KV on that card; see the exact precision and
validation scope in [Native context and Q8 KV](ROCM_NATIVE_CONTEXT.md).
Allocate a larger context explicitly and keep `--parallel 1`; fit and an
advertised capacity do not prove long-task quality.

- [Agent context validation](ROCM_AGENT_CONTEXT.md) covers bounded prefill,
  long prompts, recovery and repository-task limits. Keep tool outputs and
  instructions small even when the model advertises a large native window.
- [Native context and Q8 KV](ROCM_NATIVE_CONTEXT.md) documents opt-in
  `ZINC_KV_CACHE=q8`, its model restrictions and single-slot requirement.
  F32 KV remains the default.
- [Qwen3.8 MTP results](ROCM_MTP_0_17.md) record model-native drafting,
  controls and matched measurements. Eligible Qwen3.8 models with a NextN
  block use MTP in CLI and single-slot serving; `ZINC_MTP=0` disables it.
- [Gemma bounded prefill](ROCM_GEMMA_BOUNDED_PREFILL.md),
  [Gemma MoE prefill](ROCM_GEMMA_MOE_PREFILL.md) and
  [single-slot correctness](ROCM_SINGLE_SLOT_CORRECTNESS.md) preserve the
  model-specific investigations and regression evidence.

Check each report's source version before applying experimental controls to a
stable archive. Current performance is on the [ROCm benchmark dashboard](/zinc/benchmarks/#rdna-rocm);
[benchmarking](BENCHMARKING.md) explains the matched reusable-server method.

## Troubleshooting

| Failure                                              | What to check                                                                                                   |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| No GPU / wrong GPU                                   | Inspect `rocminfo`, apply `ROCR_VISIBLE_DEVICES`, then rerun `zinc --check`.                                    |
| `hipErrorNoBinaryForGpu`                             | Ensure the installed runtime and hipRTC support the detected `gfx*` target.                                     |
| Missing shared library or runtime compilation header | Install the matching HIP/hipRTC/hipBLAS development files and HIP/rocWMMA headers.                              |
| Custom ROCm location no longer resolves              | Use the matching toolkit's `lib` in `LD_LIBRARY_PATH`, or rebuild for its new location.                         |
| Out of memory or long-prompt failure                 | Reduce context, use one slot and inspect the model-specific context report above. Preserve logs for GPU errors. |
| JSON parser fails on a completion                    | Explicitly send `"stream": false`.                                                                              |

The standard `/opt/rocm` source build records its library location. A different
installation must make the matching libraries available to the dynamic linker.

## Build from source

Follow [Development prerequisites and backend builds](DEVELOPMENT.md#build).
Set `ROCM_PATH` to the installed toolkit root when building. ZINC's ROCm
backend shares loader/scheduler interfaces with CUDA but uses HIP, hipRTC and
hipBLAS implementations. Internal kernels and tuning switches belong in the
source reference and linked validation reports, not in the installation flow.
