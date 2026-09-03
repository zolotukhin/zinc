# Getting started with ZINC

> **Experimental software**: ZINC is under active development. The CLI path is the best-supported way to start. Server mode, model coverage, and performance tuning are still moving quickly.

ZINC runs local LLMs through Vulkan or ROCm on AMD Radeon, Vulkan on Intel Arc, and Metal on Apple Silicon. The fastest way to check if it works on your machine is:

1. Install Zig.
2. Build the binary.
3. Run one prompt from the terminal.

If that works, move on to the [hardware requirements](/zinc/docs/hardware-requirements/), [running ZINC](/zinc/docs/running-zinc/), and the lower-level tuning docs.

## Fast path

```bash
git clone https://github.com/zolotukhin/zinc.git
cd zinc
zig build -Doptimize=ReleaseFast
./zig-out/bin/zinc model pull qwen35-9b-q4k-m
./zig-out/bin/zinc --model-id qwen35-9b-q4k-m --prompt "What is the capital of France?" --chat
```

On RDNA4 Linux with the Vulkan build, set the cooperative matrix fast path before the check or first prompt:

```bash
export RADV_PERFTEST=coop_matrix
```

## Before you start

ZINC currently targets:

- **Linux** with AMD RDNA3/RDNA4 GPUs through Vulkan 1.3
- **Linux** with validated AMD RDNA4 GPUs through ROCm/HIP
- **Linux** with Intel Arc Xe2 / Battlemage GPUs through Vulkan 1.3
- **macOS** with Apple Silicon (M1 through M5) through Metal
- **Linux or WSL2** with NVIDIA RTX through the experimental CUDA backend
- **GGUF models** (Q4_K, Q5_K, Q6_K, Q8_0, Q5_0, MXFP4, F16, F32 quantizations)

**Native Windows is not supported.** The NVIDIA CUDA backend can be built experimentally under WSL2, but WSL2 is not part of the validated Vulkan or ROCm matrix.

### Models in current work

This list keeps the local ZINC models and the wider model work in one place. A model ID is present only when `zinc model pull` can install that checkpoint. Muse Spark is a hosted model, not a ZINC GGUF compatibility claim.

| Model | Runtime | Model ID or access | Fits on | Current status |
|------|------|------------|---------|--------|
| **Qwen 3.5 9B Q4_K_M** | Local GGUF | `qwen35-9b-q4k-m` · [checkpoint](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | 8+ GB VRAM or unified | AMD Vulkan/ROCm, Intel Vulkan, Apple Metal |
| **Qwen 3.6 35B-A3B Q4_K_XL** | Local GGUF | `qwen36-35b-a3b-q4k-xl` · [checkpoint](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | 24+ GB VRAM or unified | AMD Vulkan/ROCm, Intel Vulkan, Apple Metal |
| **Qwen 3.8 27B Q4_K_M** | Local GGUF | `qwen38-27b-q4k-m` · [checkpoint](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | 24+ GB VRAM or 32+ GB unified | AMD RDNA4 ROCm and Apple Metal |
| **Muse Glimmer 30B Q4_K_M** | Local GGUF | Direct file · [exact 17 GB GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/blob/main/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf) | 24+ GB VRAM or unified | AMD ROCm and Apple Metal validation target |
| **Muse Spark 1.3** | Hosted/API | [Meta release](https://research.meta.ai/blog/introducing-muse-spark-1-3) · local Muse checkpoint: [exact Glimmer GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/blob/main/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf) | Hosted | Spark open weights are forthcoming; ZINC runs its distilled Glimmer model locally |
| **Gemma 4 26B-A4B Q4_K_M** | Local GGUF | `gemma4-26b-a4b-q4k-m` · [checkpoint](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | 16+ GB VRAM or unified | AMD Vulkan/ROCm, Intel Vulkan, Apple Metal |
| **Gemma 4 31B Q4_K_M** | Local GGUF | `gemma4-31b-q4k-m` · [checkpoint](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) | 24+ GB VRAM or unified | AMD Vulkan/ROCm, Intel Vulkan, Apple Metal |

## Install dependencies

### macOS (Apple Silicon)

```bash
brew install zig
xcode-select --install
```

That is all you need. No Vulkan, no glslc, no Python, no MLX.

### Linux (AMD or Intel Arc GPU)

```bash
sudo apt update
sudo apt install -y git libvulkan-dev vulkan-tools glslc
```

Then install **Zig 0.15.2 or newer** from [ziglang.org/download](https://ziglang.org/download/).

That package set is for Vulkan. For AMD ROCm, install the HIP development files that provide `amdhip64`, `hiprtc`, and `hipblas`; the complete setup is in the [ROCm guide](/zinc/docs/rocm/).

## Clone and build

The default build selects Vulkan on Linux and Metal on macOS:

```bash
git clone https://github.com/zolotukhin/zinc.git
cd zinc
zig build -Doptimize=ReleaseFast
```

Choose ROCm explicitly on a supported AMD Linux installation:

```bash
ROCM_PATH=/opt/rocm zig build -Dbackend=rocm -Doptimize=ReleaseFast
ROCR_VISIBLE_DEVICES=0 ./zig-out/bin/zinc --check
```

The NVIDIA CUDA build is experimental:

```bash
CUDA_HOME=/usr/local/cuda zig build -Dbackend=cuda -Doptimize=ReleaseFast
./zig-out/bin/zinc --check
```

The compiled binary ends up at `./zig-out/bin/zinc`. Vulkan builds compile GLSL shaders to SPIR-V. ROCm and CUDA compile their native kernels at runtime, and macOS compiles Metal shaders from MSL source.

## Run the preflight check

Before running a prompt, verify the machine and GPU:

```bash
./zig-out/bin/zinc --check
```

On RDNA4 Linux with the Vulkan build, enable cooperative matrix first. ROCm, Intel Arc, CUDA, and macOS users skip this variable:

```bash
export RADV_PERFTEST=coop_matrix
./zig-out/bin/zinc --check
```

The check verifies GPU detection, shader assets, and runtime initialization. If it reports `READY [OK]`, you are good to go.

## Browse the model catalog

See what ZINC supports on your machine:

```bash
./zig-out/bin/zinc model list
```

The catalog auto-detects your GPU profile (`amd-rdna4-32gb`, `apple-silicon`, etc.) and shows which models fit.

## Download a model

```bash
./zig-out/bin/zinc model pull qwen35-9b-q4k-m
```

This downloads the model into a local cache and verifies the SHA-256 hash.

## Run your first prompt

The `--chat` flag wraps your prompt in the model's chat template (system prompt, role tags, etc.), which is required for instruct-tuned models to produce proper answers. Without `--chat`, the model treats the input as raw text completion, which still works but produces less focused output.

To reproduce an API chat workload exactly from the CLI, pass its system turn explicitly:

```bash
./zig-out/bin/zinc --model-id qwen38-27b-q4k-m \
  --chat \
  --system-prompt "You are a helpful assistant. Answer directly. Do not show analysis." \
  --prompt "Review this code for concurrency bugs."
```

`--system-prompt` requires both `--chat` and `--prompt`. It is useful for benchmarks because system-turn tokens are part of prefill.

**On RDNA4 Linux with Vulkan, the env var below is required** — cooperative matrix is the fast path, and without it you may see slow or incorrect output. ROCm, Intel Arc, CUDA, and macOS users skip the `export` line.

```bash
# RDNA4 Vulkan only — required, not optional. Skip for ROCm, Intel Arc, CUDA, and macOS.
export RADV_PERFTEST=coop_matrix

./zig-out/bin/zinc --model-id qwen35-9b-q4k-m --prompt "What is the capital of France?" --chat
```

Good first-run signals in the logs:

```
info(loader): Loading model: ...
info(forward): Prefill complete: ...
info(forward): Generated 256 tokens in ... ms — XX.XX tok/s
info(zinc): Output text: Paris...
```

If you see a tok/s number and coherent output, the core path is working.

## Start the chat UI

```bash
./zig-out/bin/zinc chat
```

This starts the server (default port 9090) and opens the built-in chat UI in your browser. The server also exposes an OpenAI-compatible API at `http://localhost:9090/v1`.

You can also start the server manually:

```bash
./zig-out/bin/zinc --model-id qwen35-9b-q4k-m -p 8080
```

Then open `http://localhost:8080/` in your browser.

## Manage models

```bash
# Set a default model for future runs
./zig-out/bin/zinc model use qwen35-9b-q4k-m

# Check the active default
./zig-out/bin/zinc model active

# Remove a cached model
./zig-out/bin/zinc model rm qwen35-9b-q4k-m
```

## What to read next

- [Hardware requirements](/zinc/docs/hardware-requirements/) for GPU, memory, and OS details
- [Running ZINC](/zinc/docs/running-zinc/) for CLI flags, server mode, and API endpoints
- [Serving HTTP API](/zinc/docs/api/) for the full endpoint reference
- [Development Guide](/zinc/docs/development/) for building, testing, and contributing
- [ROCm backend](/zinc/docs/rocm/) for AMD HIP setup, validation, and tuning switches
- [RDNA4 tuning](/zinc/docs/rdna4-tuning/) for AMD performance work
- [Intel GPU Reference](/zinc/docs/intel-gpu-reference/) for Arc B-series hardware details
- [Apple Silicon Reference](/zinc/docs/apple-silicon-reference/) for M1 through M5 platform details
