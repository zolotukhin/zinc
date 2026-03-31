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

### Validated models

This list is intentionally narrow. It shows the exact GGUFs that have been revalidated in ZINC, not a broader architecture wishlist.

| Model | Exact GGUF |
|------|------------|
| **Qwen3.5 2B** | [Qwen3.5-2B-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) |
| **Qwen3.5 35B-A3B UD** | [Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) |

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
zig build -Doptimize=ReleaseFast
```

The compiled binary ends up at:

```bash
./zig-out/bin/zinc
```

On Linux, `zig build -Doptimize=ReleaseFast` also compiles the GLSL shaders into SPIR-V. On macOS, shader compilation is skipped, which is one of the reasons Linux is the real runtime target.

## Run the preflight before your first prompt

Before running a full prompt, ask ZINC to validate the machine, model, and GPU budget:

```bash
# General machine + Vulkan + shader preflight
./zig-out/bin/zinc --check

# Recommended on RDNA4 shells
export RADV_PERFTEST=coop_matrix

# Check one exact GGUF file
./zig-out/bin/zinc --check -m /path/to/model.gguf

# Or check one managed model from the built-in catalog
./zig-out/bin/zinc --check --model-id qwen35-35b-a3b-q4k-xl
```

That command verifies the Vulkan path, required shader assets, and the current single-GPU VRAM fit estimate. When you pass `-m`, it checks the exact GGUF file. When you pass `--model-id`, it checks the managed catalog entry by name. If it reports `NOT READY [FAIL]`, fix that first before trying prompt generation.

## Inspect the managed model catalog

If you want to see what ZINC currently marks as supported on the local machine:

```bash
# Show models that are both tested for the detected GPU profile
# and estimated to fit the current VRAM budget
./zig-out/bin/zinc model list

# Show the full built-in catalog even if Vulkan is unavailable locally
./zig-out/bin/zinc model list --all
```

If you want ZINC to manage downloads and default model selection for you:

```bash
# Download one managed model into the local cache
./zig-out/bin/zinc model pull qwen35-2b-q4k-m

# Mark it as the active default for future runs
./zig-out/bin/zinc model use qwen35-2b-q4k-m

# Inspect the current managed default
./zig-out/bin/zinc model active
```

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
# Check the local toolchain and Vulkan stack
zig version
glslc --version
vulkaninfo --summary

# Ask ZINC for a general readiness check
./zig-out/bin/zinc --check

# Ask ZINC about one exact GGUF file
./zig-out/bin/zinc --check -m /path/to/model.gguf
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
