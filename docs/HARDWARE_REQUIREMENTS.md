# ZINC hardware requirements

> **Experimental and Linux-first**: ZINC is under active development. The most reliable environment today is Linux on AMD RDNA3 or RDNA4 with Vulkan 1.3.

This page answers the practical question first: what hardware and OS setup do you need to actually run ZINC without fighting the stack?

## Recommended target environment

The current sweet spot is:

- **OS**: Linux
- **GPU**: AMD RDNA3 or RDNA4
- **API**: Vulkan 1.3
- **Drivers**: RADV or AMDVLK
- **Model format**: GGUF

If you match that stack, you are inside the environment ZINC is tuned and tested around.

## GPU support

ZINC is being built specifically for AMD consumer and workstation GPUs that the mainstream ROCm-first stack underserves.

### Primary target GPUs

| Family | Examples | Status |
| --- | --- | --- |
| RDNA4 | RX 9070, RX 9070 XT, Radeon AI PRO R9700 | Primary tuning target |
| RDNA3 | RX 7900 XTX, RX 7900 XT, RX 7800 XT, RX 7700 XT, RX 7600 | Supported direction, less tuned than RDNA4 |

### Experimental or secondary paths

| Family | Status |
| --- | --- |
| Intel Arc | Possible through Vulkan, not a primary target |
| NVIDIA via Vulkan | Not a primary target |
| macOS GPUs | Build environment only, not the practical runtime target |

## Vulkan requirements

You need a working Vulkan stack that can see the GPU:

```bash
vulkaninfo --summary
```

If that command does not show your AMD GPU, ZINC will not work yet no matter what model or command line you use.

## Fastest preflight command

Once the binary is built, the quickest end-to-end sanity check is:

```bash
# General machine + Vulkan + shader preflight
./zig-out/bin/zinc --check

# Check one exact GGUF file
./zig-out/bin/zinc --check -m /path/to/model.gguf

# Or check one managed model from the built-in catalog
./zig-out/bin/zinc --check --model-id qwen35-35b-a3b-q4k-xl
```

That command combines the practical questions you usually care about:

- does the Vulkan stack initialize?
- did ZINC find the intended GPU?
- are the runtime shader assets present?
- is the GGUF file valid when you pass `-m`?
- does the managed catalog entry look runnable here when you pass `--model-id`?
- does the current single-GPU runtime appear to fit into device-local VRAM?

For RDNA4 shells, use:

```bash
# Recommended on RDNA4 shells
export RADV_PERFTEST=coop_matrix

# Check one exact GGUF file
./zig-out/bin/zinc --check -m /path/to/model.gguf
```

If you want to see which managed models ZINC currently marks as supported on the machine, use:

```bash
# Tested for this GPU profile and estimated to fit current VRAM
./zig-out/bin/zinc model list

# Full catalog, even if local GPU probing is unavailable
./zig-out/bin/zinc model list --all
```

## Driver and runtime notes

For the best current path on AMD Linux:

- use a working **Vulkan 1.3** stack
- use **RADV** or **AMDVLK**
- on RDNA4, set `RADV_PERFTEST=coop_matrix`

Typical RDNA4 runtime setup:

```bash
export RADV_PERFTEST=coop_matrix
# Optional: append --debug or use ZINC_DEBUG=1 for diagnostic logs
./zig-out/bin/zinc -m /path/to/model.gguf --prompt "hello"
```

The benchmark node behind ZINC uses Linux with Mesa/RADV and an RDNA4 GPU. That is the environment the project’s tuning notes are grounded in.

## VRAM expectations

The right question is not just “does the GPU work?” but “does the GPU have enough VRAM for the model and context you want?”

### Practical guide

| VRAM | What it is good for |
| --- | --- |
| 16 GB | Practical for smaller GGUF models and experiments, especially 7B to 8B class models |
| 32 GB | The current ZINC development target for larger models such as Qwen3.5-35B-A3B Q4_K_XL |

Exact fit depends on:

- model architecture
- quantization
- context length
- whether KV compression is enabled
- whether you are serving multiple sessions

`--check -m ...` now prints a practical fit estimate for the current engine. The most important lines are:

- `Tensor upload`: exact device-local weight bytes derived from GGUF tensors
- `VRAM fit`: estimated device-local total versus the selected GPU's VRAM budget
- `KV cache`: the current runtime estimate, with today's engine capped to a `4096` token KV budget

Treat that estimate as operational guidance, not a formal allocator proof. It intentionally does not include every source of overhead, such as Vulkan allocation alignment, descriptor pools, query pools, or driver-side bookkeeping.

If you are just getting started, treat **16 GB** as the lower comfortable floor and **32 GB** as the better target for the larger-model work shown in the repo.

## System RAM and CPU

ZINC is GPU-centric, but the machine still needs enough system resources to stage model load and run the process cleanly.

### Recommended baseline

- **CPU**: modern x86_64 CPU
- **System RAM**: 32 GB minimum, 64 GB preferred for larger models and smoother experimentation
- **Storage**: NVMe SSD strongly recommended

You do not need a server CPU. You do need a machine stable enough to load multi-gigabyte GGUF files and keep the Vulkan stack happy.

## OS support

### Linux

This is the real target today. If you want to use ZINC, start here.

### macOS

Useful for editing, building, and some development workflows. Not the environment to judge actual ZINC GPU inference.

## How to sanity-check your machine

```bash
lspci | rg -i "vga|display|amd|radeon"
vulkaninfo --summary
free -h
./zig-out/bin/zinc --check -m /path/to/model.gguf
```

Those four commands answer most early compatibility questions:

- did Linux see the GPU?
- did Vulkan see the GPU?
- does the machine have enough memory to stage the workload?
- does ZINC think the current runtime and model fit together?

## If you want the shortest path to success

Use:

- Linux
- AMD RDNA4
- Vulkan 1.3
- GGUF models
- CLI mode first

Then move on to [Running ZINC](/zinc/docs/running-zinc) and [RDNA4 tuning](/zinc/docs/rdna4-tuning).
