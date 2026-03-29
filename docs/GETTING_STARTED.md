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
- **GGUF models** (Q4_K, Q5_K, Q6_K, Q8_0, F16 quantizations)
- developers who want to run CLI inference first, then experiment with serving

### Supported models

| Architecture | Example models |
|-------------|---------------|
| Qwen3.5 MoE (hybrid SSM+attention) | [Qwen3.5-35B-A3B](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) |
| Qwen3 / Qwen2 MoE | [Qwen3-30B-A3B](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF), [Qwen2.5-32B](https://huggingface.co/unsloth/Qwen2.5-32B-Instruct-GGUF) |
| LLaMA / Mistral | [LLaMA 3.1 8B](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF), [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.3) |
| Mamba / Jamba (SSM) | [Jamba-v0.1](https://huggingface.co/ai21labs/Jamba-v0.1) |

The primary test model is **Qwen3.5-35B-A3B** — download [Q4_K_XL from unsloth](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) (21 GB, fits in 32 GB VRAM).

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

## Start the chat interface

Once CLI mode works, start the server to get a ChatGPT-like web UI:

```bash
export RADV_PERFTEST=coop_matrix
# Optional: append --debug or use ZINC_DEBUG=1 for diagnostic logs
./zig-out/bin/zinc -m /path/to/model.gguf -p 8080
```

Then open **http://localhost:8080/** in your browser. The chat interface is built into the ZINC binary — no separate install needed.

The server also exposes an OpenAI-compatible API at `http://localhost:8080/v1` that works with any OpenAI SDK client.

## What to read next

- [Running ZINC](/zinc/docs/running-zinc) for CLI flags, server mode, API endpoints, and SDK examples
- [Hardware requirements](/zinc/docs/hardware-requirements) for GPU, VRAM, and OS expectations
- [Serving HTTP API](/zinc/docs/api) for the full endpoint reference
- [RDNA4 tuning](/zinc/docs/rdna4-tuning) if you are chasing performance on Linux
