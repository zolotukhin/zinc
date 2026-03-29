# Getting started with ZINC

> **Experimental software**: ZINC is still under active development. The CLI path is the best-supported way to use it today. Server mode, model coverage, and performance tuning are still moving quickly.

ZINC is a local LLM inference engine for AMD GPUs built in Zig and Vulkan. The fastest way to understand whether it fits your machine is:

1. Make sure you are on **Linux** with a supported AMD Vulkan stack.
2. Build the binary.
3. Run one GGUF model with one prompt from the terminal.

If that works, then move on to the [hardware requirements](/zinc/docs/hardware-requirements), [running ZINC](/zinc/docs/running-zinc), and the lower-level tuning docs.

## Before you start

ZINC is currently aimed at:

- **Linux** for real GPU inference
- **AMD RDNA3 / RDNA4 GPUs** through Vulkan 1.3
- **GGUF models**
- developers who want to run CLI inference first, then experiment with serving

macOS can build the project, but it is not the target environment for actual ZINC GPU inference. If you want the shortest path to success, use Linux on AMD hardware.

## Install the basic dependencies

You need:

- **Zig 0.15.2+**
- **Vulkan loader and headers**
- **`glslc`** for shader compilation on Linux
- **Git**

On Ubuntu or Debian, the system packages are a good starting point for the Vulkan toolchain:

```bash
sudo apt update
sudo apt install -y git libvulkan-dev vulkan-tools glslc
```

Then install **Zig 0.15.2 or newer** from the official Zig downloads page:

- [ziglang.org/download](https://ziglang.org/download/)

## Clone and build

```bash
git clone https://github.com/zolotukhin/zinc.git
cd zinc
zig build
```

The compiled binary ends up at:

```bash
./zig-out/bin/zinc
```

On Linux, `zig build` also compiles the GLSL shaders into SPIR-V. On macOS, shader compilation is skipped, which is one of the reasons Linux is the real runtime target.

## Run your first prompt

On RDNA4, enable cooperative matrix support before running:

```bash
export RADV_PERFTEST=coop_matrix
./zig-out/bin/zinc \
  -m /path/to/model.gguf \
  --prompt "The capital of France is"
```

Good first-run signals in the logs look like this:

```bash
info(vulkan): Selected GPU 0: AMD Radeon Graphics
info(loader): Loading model: /path/to/model.gguf
info(forward): Prefill complete: ...
info(forward): Generated ... tok/s
```

If you get through model load, prefill, and at least one decode step, the core path is working.

## Verify the environment quickly

These commands are useful before blaming ZINC:

```bash
zig version
glslc --version
vulkaninfo --summary
```

If `vulkaninfo` does not see your AMD GPU, fix that first. ZINC sits on top of the Vulkan stack you already have.

## What to read next

- [Hardware requirements](/zinc/docs/hardware-requirements) for GPU, VRAM, and OS expectations
- [Running ZINC](/zinc/docs/running-zinc) for CLI flags, server mode, and graph exports
- [Serving HTTP API](/zinc/docs/api) if you want the OpenAI-compatible endpoint shape
- [RDNA4 tuning](/zinc/docs/rdna4-tuning) if you are chasing performance on Linux

## Current reality

ZINC is not positioned as a finished plug-and-play runtime yet. It is a fast-moving systems project with a working inference core, evolving serving path, and aggressive hardware-specific tuning. If you treat it as experimental software and start from the CLI, the experience is much better.
