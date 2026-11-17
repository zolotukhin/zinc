# Hardware and memory

Choose a GPU/backend using the table below, then follow
[installation and first run](GETTING_STARTED.md). This page distinguishes
published release qualification from broader source support.

## Supported platforms

| Platform and GPU                      | Backend               | Evidence and limits                                                                         |
| ------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------- |
| Linux x86_64, AMD Radeon AI PRO R9700 | Vulkan or ROCm        | Both have stable 0.1.1 archives and qualified model/context rows.                           |
| Linux x86_64, Intel Arc Pro B70 / Xe2 | Vulkan                | Stable 0.1.1 qualification; other Arc cards need their own fit and driver checks.           |
| Other AMD RDNA3/RDNA4 and Strix Halo  | Vulkan source support | Outside the stable launch matrix. Source compatibility does not qualify every device/model. |
| Apple Silicon M1–M5                   | Metal                 | Source backend and unsigned preview; no stable 0.1.1 Mac binary.                            |
| NVIDIA RTX 4090/5090, Linux or WSL2   | CUDA                  | Experimental source backend; outside stable qualification.                                  |

The [release requirements](GETTING_STARTED.md#choose-your-backend) specify OS/ABI
floors. The [published validation](/zinc/validation/release-candidate-0.1.1-2026-11-16.json)
records tested drivers, model files, contexts and precision. An OS meeting an
archive's glibc floor does not establish GPU driver compatibility.

## Driver checks

Run the check for your backend, then `zinc --check`:

| Backend       | Check                                | What must be present                                                                             |
| ------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------ |
| AMD Vulkan    | `vulkaninfo --summary`               | The intended AMD GPU and Vulkan 1.3 driver; RADV is the primary test path.                       |
| Intel Vulkan  | `vulkaninfo --summary`               | The intended Intel Arc GPU with ANV/Mesa. Enable Resizable BAR for comparable performance.       |
| ROCm          | `rocminfo`                           | A supported native `gfx*` target; matching HIP/hipRTC/hipBLAS libraries and HIP/rocWMMA headers. |
| Metal preview | `system_profiler SPDisplaysDataType` | Native Apple Silicon; preview archives require macOS 15+ and the Mac's factory OS minimum.       |
| CUDA source   | `nvidia-smi`                         | A working NVIDIA driver plus the CUDA Driver API, NVRTC, cuBLAS and runtime libraries.           |

```bash
zinc --version
zinc --check
```

Vulkan and ROCm use independent device lists and driver stacks. Set
`RADV_PERFTEST=coop_matrix` only for AMD RDNA4 Vulkan. For ROCm runtime setup,
GPU selection and failures, use the [ROCm guide](ROCM.md). Missing a compiler
is not a reason to install development tools for a prebuilt Vulkan archive.

## Model memory

Model weights, KV cache, runtime scratch and other GPU users share the memory
budget. A parameter count alone is not a reliable fit test. On Apple Silicon,
the OS and applications also share unified memory with the GPU.

```bash
zinc model list
zinc model list --all
zinc --check --model-id qwen35-9b-q4k-m -c 4096
```

The default list filters by tested GPU profile and estimated fit. `--all` shows
entries that may be unsupported or too large; it is not a list of models you
can necessarily run. For an exact local file, use `zinc --check -m
/path/to/model.gguf -c 4096`.

Start at 4K context for chat. Client guides use 16K for tools and history, which
requires more memory. The 32 GB R9700 and Arc Pro B70 are the recorded large-model
targets; do not infer the same capacity for a smaller card or a Mac with the
same total memory. Increase context or request slots only after checking fit
and successfully running the intended workload. Preflight estimates can omit
lazy allocations and driver overhead.

Use an SSD with enough free space for each GGUF and future downloads. Reserve
system RAM for the OS, tokenizer and host allocations as well as GPU memory.
For agent work, consult the [OpenClaw](OPENCLAW.md#tested-scope) and
[OpenCode](OPENCODE.md#tested-scope) task results: a model that fits or generates
text may still fail file edits and tests.

## Hardware and source references

- [AMD GPU reference](AMD_GPU_REFERENCE.md) and [RDNA4 tuning](RDNA4_TUNING.md).
- [Intel GPU reference](INTEL_GPU_REFERENCE.md) and [matched benchmarks](/zinc/benchmarks/#intel).
- [Apple Silicon reference](APPLE_SILICON_REFERENCE.md) and [Metal reference](APPLE_METAL_REFERENCE.md).
- [NVIDIA GPU reference](NVIDIA_GPU_REFERENCE.md).
- [Build from source](DEVELOPMENT.md#build), including experimental CUDA.
