# Getting started with ZINC

> **Experimental software**: ZINC is under active development. The CLI path is the best-supported way to start. Server mode, model coverage, and performance tuning are still moving quickly.

ZINC is a local LLM inference engine for AMD GPUs and Apple Silicon. The fastest way to check if it works on your machine:

1. Install Zig.
2. Build the binary.
3. Run one prompt from the terminal.

If that works, move on to the [hardware requirements](/zinc/docs/hardware-requirements), [running ZINC](/zinc/docs/running-zinc), and the lower-level tuning docs.

## Before you start

ZINC currently targets:

- **Linux** with AMD RDNA3/RDNA4 GPUs through Vulkan 1.3
- **macOS** with Apple Silicon (M1 through M5) through Metal
- **GGUF models** (Q4_K, Q5_K, Q6_K, Q8_0, F16 quantizations)

### Supported models

This list is intentionally narrow. It shows the exact GGUFs that have been validated end-to-end.

| Model | Exact GGUF | Fits on |
|------|------------|---------|
| **Qwen3.5 2B** | [Qwen3.5-2B-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) | 16+ GB VRAM or unified |
| **Qwen3.5 35B-A3B UD** | [Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) | 24+ GB VRAM or unified |

## Install dependencies

### macOS (Apple Silicon)

```bash
brew install zig
xcode-select --install
```

That is all you need. No Vulkan, no glslc, no Python, no MLX.

### Linux (AMD GPU)

```bash
sudo apt update
sudo apt install -y git libvulkan-dev vulkan-tools glslc
```

Then install **Zig 0.15.2 or newer** from [ziglang.org/download](https://ziglang.org/download/).

## Clone and build

Same on both platforms:

```bash
git clone https://github.com/zolotukhin/zinc.git
cd zinc
zig build -Doptimize=ReleaseFast
```

The compiled binary ends up at `./zig-out/bin/zinc`. On Linux, the build also compiles GLSL shaders to SPIR-V. On macOS, Metal shaders are compiled at runtime from MSL source.

## Run the preflight check

Before running a prompt, verify the machine and GPU:

```bash
./zig-out/bin/zinc --check
```

On RDNA4 Linux, enable cooperative matrix first:

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
./zig-out/bin/zinc model pull qwen35-2b-q4k-m
```

This downloads the model into a local cache and verifies the SHA-256 hash.

## Run your first prompt

```bash
./zig-out/bin/zinc --model-id qwen35-2b-q4k-m --prompt "The capital of France is"
```

On RDNA4 Linux, remember to set the environment variable:

```bash
export RADV_PERFTEST=coop_matrix
./zig-out/bin/zinc --model-id qwen35-2b-q4k-m --prompt "The capital of France is"
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
./zig-out/bin/zinc --model-id qwen35-2b-q4k-m -p 8080
```

Then open `http://localhost:8080/` in your browser.

## Manage models

```bash
# Set a default model for future runs
./zig-out/bin/zinc model use qwen35-2b-q4k-m

# Check the active default
./zig-out/bin/zinc model active

# Remove a cached model
./zig-out/bin/zinc model rm qwen35-2b-q4k-m
```

## What to read next

- [Hardware requirements](/zinc/docs/hardware-requirements) for GPU, memory, and OS details
- [Running ZINC](/zinc/docs/running-zinc) for CLI flags, server mode, and API endpoints
- [Serving HTTP API](/zinc/docs/api) for the full endpoint reference
- [Development Guide](/zinc/docs/development) for building, testing, and contributing
- [RDNA4 tuning](/zinc/docs/rdna4-tuning) for AMD performance work
- [Apple Silicon Reference](/zinc/docs/apple-silicon-reference) for M1 through M5 platform details
