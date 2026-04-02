<p align="center">
  <img src="assets/zinc_trademark.png" alt="ZINC Logo" width="400">
</p>

# ZINC — Zig INferenCe Engine for AMD GPUs

<p align="center">
  <a href="https://github.com/zolotukhin/zinc/actions/workflows/test.yml">
    <img src="https://github.com/zolotukhin/zinc/actions/workflows/test.yml/badge.svg" alt="CI Status">
  </a>
  <a href="https://ziglang.org/download/">
    <img src="https://img.shields.io/badge/Zig-0.15.2-orange.svg?logo=zig&logoColor=white" alt="Zig Version">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  </a>
  <img src="https://img.shields.io/badge/platform-linux-lightgrey" alt="Platform">
  <a href="https://zolotukhin.ai/zinc">
    <img src="https://img.shields.io/badge/web-zolotukhin.ai%2Fzinc-8B5CF6" alt="Website">
  </a>
  <a href="https://discord.gg/tNDEgTG5s">
    <img src="https://img.shields.io/badge/Discord-Join%20ZINC-5865F2?logo=discord&logoColor=white" alt="ZINC Discord">
  </a>
</p>

> Making AMD consumer GPUs actually usable for LLM inference.

<p align="center">
  <img src="assets/zinc-chat-demo.gif" alt="ZINC Chat Demo — streaming inference on AMD RDNA4" width="720">
  <br>
  <em>35B parameter model running locally on a single AMD GPU — Zig + Vulkan, no ROCm, no CUDA</em>
</p>

## Best Supported Setup

- Linux
- AMD RDNA4 GPU with 16-32 GB VRAM
- Mesa RADV with `RADV_PERFTEST=coop_matrix`
- `zig build -Doptimize=ReleaseFast`

macOS can build the project and is fine for development, but it is not the primary target environment for real ZINC GPU inference.

## Start Here

If you want the shortest path to a successful first run:

```bash
git clone https://github.com/zolotukhin/zinc.git
cd zinc
zig build -Doptimize=ReleaseFast

# Recommended on RDNA4 before checks and benchmarks
export RADV_PERFTEST=coop_matrix

# Verify Vulkan, shaders, and runtime setup
./zig-out/bin/zinc --check

# See which supported models fit this machine
./zig-out/bin/zinc model list

# Download one supported model
./zig-out/bin/zinc model pull qwen35-2b-q4k-m

# Run a prompt with the managed model
./zig-out/bin/zinc --model-id qwen35-2b-q4k-m --prompt "Hello"
```

If you want the browser UI instead, start the server with:

```bash
./zig-out/bin/zinc --model-id qwen35-2b-q4k-m -p 8080
```

Then open `http://localhost:8080/` for the built-in chat UI. The server also exposes an OpenAI-compatible API at `http://localhost:8080/v1` — use it with any client that supports the OpenAI chat completions format.

## What Works Today

- Single-stream CLI inference on the validated Qwen3.5 models listed below
- OpenAI-compatible `/v1` API
- Built-in browser chat UI at `/`
- Managed model workflow: `list`, `pull`, `use`, `active`, and `rm`
- RDNA4-tuned Vulkan path with coherent outputs on the supported GGUFs

## Still Rough

- Continuous batching and stronger multi-tenant serving are still roadmap work
- Chat/reasoning workloads are still slower than the raw decode path
- The supported-model list is intentionally narrow
- macOS is useful for building and docs work, but not the target runtime for GPU inference

## The Problem

AMD's RDNA3/RDNA4 GPUs (RX 9070, Radeon AI PRO R9700, etc.) have excellent memory bandwidth (576+ GB/s) and hardware features (cooperative matrix, integer dot product), but:

1. **ROCm doesn't support them** — only MI-series datacenter GPUs
2. **vLLM requires ROCm** — so it can't use these GPUs at all
3. **llama.cpp Vulkan works** but treats RDNA4 as an afterthought — no RDNA4-specific tuning, SPIR-V toolchain incompatibilities, no tensor parallelism
4. **No solution handles parallel requests well** on these GPUs for production use

These cards cost $500–1500 (vs $15,000+ for MI300X) and sit in millions of desktops doing nothing during inference.

## The Solution

ZINC takes the hardware these cards already have — 576 GB/s memory bandwidth, cooperative matrix units, 16–32 GB VRAM — and builds an inference engine that actually uses it.

**Hand-tuned for the hardware.** The GPU shaders are written specifically for RDNA4's memory hierarchy: wave64 dispatch, architecture-aware tiling, fused operations that cut redundant VRAM round-trips. Not a generic Vulkan backend that happens to run on AMD — built to hit 90%+ of theoretical memory bandwidth on the matmuls that dominate LLM decode.

**Built for real inference work, not just demos.** The current engine already has a fast CLI path, an OpenAI-compatible API, graph-report tooling, and hardware-aware benchmarking on RDNA4. Continuous batching and deeper TurboQuant validation are still roadmap work, so today's server path should be read as a strong single-stream engine rather than a finished multi-tenant serving stack.

**Drop-in compatible.** The API is OpenAI-compatible — point your existing client at it and it works. No ROCm, no CUDA, no driver stack to fight. One binary, one GPU, production inference on a $550 card.

## Supported Models

The table below is intentionally narrow: it lists the exact GGUFs ZINC currently supports and that we have revalidated end-to-end, not a broader wishlist of architectures that might work.

| Model | Exact GGUF tested | Typical throughput on AI PRO R9700 |
|------|--------------------|-------------------------------------|
| **Qwen3.5 2B** | [Qwen3.5-2B-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) | ~27 tok/s plain generation |
| **Qwen3.5 35B-A3B UD** | [Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) | ~38 tok/s plain generation |

Benchmark details for the numbers above:

- Hardware: AMD Radeon AI PRO R9700 (RDNA4, 32 GB)
- Build: `zig build -Doptimize=ReleaseFast`
- Run shape: single-stream, `RADV_PERFTEST=coop_matrix`
- Plain generation: 256-token runs without chat template or explicit reasoning
- Reasoning chat: non-streaming `/v1/chat/completions` with step-by-step prompts
- Latest validation date: 2026-03-31
- Validation: coherent output on CLI, raw `/v1/completions`, and `/v1/chat/completions`

**Quantization formats implemented in the current kernels**: Q4_K, Q5_K, Q6_K, Q8_0, F16

## Quick Start

### Prerequisites

| Tool | Install |
|------|---------|
| Zig 0.15.2+ | [ziglang.org/download](https://ziglang.org/download/) |
| Vulkan loader + tools | `apt install libvulkan-dev vulkan-tools` (Linux) or `brew install vulkan-loader vulkan-headers` (macOS) |
| `glslc` on Linux | `apt install glslc` |
| Bun for tests and the docs site | `curl -fsSL https://bun.sh/install \| bash` |

**Important**: On Linux with RDNA4, newer `glslc` releases can cause a large regression. Use the system package version.

### Build ZINC

```bash
git clone https://github.com/zolotukhin/zinc.git
cd zinc

# Build the CLI and server
# macOS: shaders are skipped
# Linux: shaders are compiled automatically
zig build -Doptimize=ReleaseFast
```

The binary is placed in `zig-out/bin/zinc`. Compiled SPIR-V shaders go to `zig-out/share/zinc/shaders/`.
Use `ReleaseFast` for any performance measurement or server deployment. Plain `zig build` is not a fair throughput baseline.

### Run a Preflight Check First

Before your first prompt, run `--check`. The target state is a clean `READY [OK]` run with no warnings.

```bash
# General machine + Vulkan + shader preflight
./zig-out/bin/zinc --check

# Recommended on RDNA4 before measuring performance
export RADV_PERFTEST=coop_matrix
./zig-out/bin/zinc --check

# Check one exact GGUF file
./zig-out/bin/zinc --check -m /path/to/model.gguf

# Check one managed catalog model by id
./zig-out/bin/zinc --check --model-id qwen35-35b-a3b-q4k-xl
```

`--check` verifies:

- host environment and RDNA4-specific shell hints
- compiled shader assets
- Vulkan device discovery and the selected GPU
- GGUF metadata when you pass `-m /path/to/model.gguf`
- managed-model compatibility when you pass `--model-id <id>`
- estimated single-GPU VRAM fit for the current runtime

If `--check` reports warnings, treat them as setup work to finish before judging runtime behavior. For the full walkthrough, see [Running ZINC](docs/RUNNING_ZINC.md) and [Hardware requirements](docs/HARDWARE_REQUIREMENTS.md).

### Choosing Models

The README keeps the supported-model table narrow on purpose and leaves the full managed-model workflow to the docs.

Use these for model selection, cache management, and API details:

- [Running ZINC](https://zolotukhin.ai/zinc/docs/running-zinc)
- [Serving HTTP API](https://zolotukhin.ai/zinc/docs/api)

### Run a Prompt

```bash
./zig-out/bin/zinc -m /path/to/model.gguf --prompt "The capital of France is"
```

### Run the Server

Start the server — no `--prompt` flag means server mode:

```bash
./zig-out/bin/zinc -m /path/to/model.gguf -p 8080
```

Then open **http://localhost:8080/** in your browser for the built-in chat interface.

### Use the API

ZINC exposes an OpenAI-compatible API at `/v1`.

For the actual request examples and SDK usage, use the website docs instead of the README:

- [Running ZINC](https://zolotukhin.ai/zinc/docs/running-zinc) for CLI, server mode, and first-run examples
- [Serving HTTP API](https://zolotukhin.ai/zinc/docs/api) for `curl`, OpenAI SDK examples, endpoint behavior, and response shapes

The built-in chat UI is served at `/`, the API is under `/v1`, and the health endpoint is `/health`.

## Development

For building, testing, debugging, benchmarking, graph export, and contributing — see the **[Development Guide](./docs/DEVELOPMENT.md)** ([web version](https://zolotukhin.ai/zinc/docs/development)).

Quick start:

```bash
zig build -Doptimize=ReleaseFast   # build
zig build test                      # run all tests
./zig-out/bin/zinc --check          # verify GPU/runtime setup
```

See also: [CONTRIBUTING.md](./CONTRIBUTING.md) · [Code of Conduct](./CODE_OF_CONDUCT.md)

## Architecture

<p align="center">
  <img src="assets/architecture.svg" alt="ZINC Architecture" width="680">
</p>

## Benchmarks

All numbers below were measured on **AMD Radeon AI PRO R9700** (RDNA4, 32 GB, 576 GB/s) on a clean RDNA4 node using `RADV_PERFTEST=coop_matrix` and `zig build -Doptimize=ReleaseFast`.

### Current Validated Snapshot (2026-03-31)

| Path | Shape | Result |
|------|-------|--------|
| Qwen3.5-35B-A3B-UD CLI plain decode | `--prompt "The capital of France is"`; 128 generated tokens | **37.95 tok/s**, `26.3 ms/tok` |
| Qwen3.5-2B-Q4_K_M CLI plain decode | `--prompt "The capital of France is"`; 128 generated tokens | **26.71 tok/s**, `37.4 ms/tok` |

For reference, the current llama.cpp baseline on the same node and 35B model is about **107 tok/s decode**.

### What These Numbers Mean

- The clean ReleaseFast decode path is approaching `40 tok/s` on the 35B model.
- The 2B model is currently slower than the 35B MoE model on this node, which means today's bottleneck is not just "smaller model = faster"; kernel shape, architecture mix, and decode-path efficiency matter more than parameter count alone.

### Why GPU Bandwidth Is Still Not "Full"

At `33.58 tok/s`, the modeled full-token decode bandwidth is about **112.5 GB/s**, or **19.5%** of the card's `576 GB/s` peak.

That is not a contradiction. Single-stream decode is not a pure DRAM-streaming workload. The remaining headroom is dominated by serialized medium/small kernels and graph depth, not by large host-side stalls. If the goal is to drive memory bandwidth materially higher than this, the next lever is **concurrent decode / batching**, not expecting one stream to saturate all DRAM bandwidth on its own.

### Historical Note

The older March 27–29 optimization logs in `.zinc_optimize/` were useful for correctness and early performance work, but many of the old `7–16 tok/s` figures came from debug-heavy or non-`ReleaseFast` builds. The snapshot above is the current clean baseline to compare against.

## Current Status

| Component | Status |
|-----------|--------|
| Vulkan infrastructure | Done |
| GGUF parser + model loader | Done |
| GPU detection (RDNA3/4) | Done |
| Native BPE tokenizer (from GGUF) | Done |
| GLSL compute shaders (16) | Done |
| Compute graph + architecture builders | Done |
| Forward pass (decode loop) | Working — 33.58 tok/s clean CLI on Qwen3.5-35B-A3B-UD |
| GPU SSM shaders + cmd batching | Done — clean ReleaseFast path is above 30 tok/s |
| HTTP server + OpenAI API | Done — 35B raw API ~33.5 tok/s, 2B raw API ~21.9 tok/s, reasoning chat still slower |
| Continuous batching | Phase 4 |
| TurboQuant KV compression | Phase 5 |

Validated on AMD Radeon AI PRO R9700 (RDNA4): Vulkan 1.3 init, GGUF parsing, 21 GB model loaded to VRAM, 723-node MoE graph built, coherent inference output verified against CPU reference.

## Next Steps

The next push is from "raw decode above 30" to "reasoning workloads above 30 and better aggregate GPU utilization":

1. **Close the chat/reasoning gap** — benchmark longer chat prompts, template overhead, stop behavior, and TTFT so `/v1/chat/completions` tracks closer to the raw decode path.
2. **Make profiling representative** — `--profile` is still too intrusive in `ReleaseFast`, so it is not yet the right leaderboard tool for apples-to-apples throughput claims.
3. **Reduce hot-path descriptor churn** — reuse bindings and trim per-token Vulkan setup in the decode loop.
4. **Tune the actual hot shapes** — focus on medium/small decode kernels, not just the vocab projection.
5. **Increase aggregate throughput with batching** — if the goal is to drive bandwidth utilization much higher, concurrency is the right lever.

## License

MIT
