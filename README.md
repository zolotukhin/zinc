<p align="center">
  <img src="assets/zinc_trademark_new.png" alt="ZINC Logo" width="400">
</p>

# ZINC — Zig INferenCe Engine

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
  <img src="https://img.shields.io/badge/platform-linux%20%7C%20macOS-lightgrey" alt="Platform">
  <a href="https://zolotukhin.ai/zinc">
    <img src="https://img.shields.io/badge/web-zolotukhin.ai%2Fzinc-8B5CF6" alt="Website">
  </a>
  <a href="https://discord.gg/QRUgWH2aGV">
    <img src="https://img.shields.io/badge/Discord-Join%20ZINC-5865F2?logo=discord&logoColor=white" alt="ZINC Discord">
  </a>
</p>

> Fastest measured local LLM inference for AMD GPUs. ZINC beats llama.cpp across the current five-model RDNA4 headline sweep — decode, prefill, end-to-end, and model-level overall — with no ROCm.

<p align="center">
  <img src="assets/amd-rdna4-benchmark-sweep.svg" alt="AMD RDNA4 benchmark sweep chart showing ZINC ahead of llama.cpp across five models" width="860">
</p>

## AMD RDNA4: ZINC Beats llama.cpp

ZINC is the fastest measured local AI engine for AMD GPUs in our current suite.
On the Radeon AI PRO R9700, it beats llama.cpp on all five published models:
decode, prefill, end-to-end, and overall.

Same GPU. Same GGUF files. Same prompts. Same server-vs-server harness.

| Model | Decode | Prefill | Overall |
|-------|-------:|--------:|--------:|
| Qwen 3.6 35B A3B UD Q4_K_XL | **166.8** vs 108.5 tok/s (**1.54x**) | **540** vs 397 tok/s (**1.36x**) | **151%** |
| Qwen 3.5 9B Q4_K_M | **97.5** vs 85.5 tok/s (**1.14x**) | **739** vs 549 tok/s (**1.35x**) | **115%** |
| Qwen 3.6 27B Dense Q4_K_M | **32.0** vs 30.7 tok/s (**1.04x**) | **213** vs 184 tok/s (**1.16x**) | **105%** |
| Gemma 4 26B-A4B MoE Q4_K_M | **113.7** vs 102.1 tok/s (**1.11x**) | **809** vs 497 tok/s (**1.63x**) | **115%** |
| Gemma 4 31B Q4_K_M | **28.8** vs 28.5 tok/s (**1.01x**) | **249** vs 200 tok/s (**1.25x**) | **103%** |

The closest row is Gemma 4 31B decode at `1.01x`. We are still cooking: the next work is widening that margin and adding more scenarios.

## Supported Platforms

| Platform | GPU | Backend | Status |
|----------|-----|---------|--------|
| **Linux** | AMD RDNA4 (RX 9070, AI PRO R9700) | Vulkan | Primary — hand-tuned shaders |
| **Linux** | AMD RDNA3 (RX 7900 XTX, etc.) | Vulkan | Supported |
| **Linux** | Intel Arc Xe2 / Battlemage | Vulkan | Supported — validated benchmark target |
| **macOS** | Apple Silicon (M1, M2, M3, M4, M5) | Metal | Supported — native MSL shaders |

ZINC focuses on current local-inference models people are actively running:
Qwen 3.5/3.6 and Gemma 4 today, with a managed catalog that stays narrow on
purpose. Older Llama/Mistral/Gemma generations may work eventually, but broad
legacy-model coverage is not the main optimization target.

## Status vs llama.cpp

Latest checked-in benchmark artifact, same machine, same weights, same prompt:

| Platform | Compared models | Decode vs llama.cpp | Prefill vs llama.cpp | Read this as |
|----------|----------------:|--------------------:|---------------------:|--------------|
| AMD RDNA4 / Vulkan | 5 | 117% avg, 5/5 model wins | 135% avg, 5/5 model wins | Clean current sweep: every published RDNA model is ahead on decode, prefill, end-to-end, and model-level overall |
| Intel Arc / Vulkan | 5 | 103% avg, 5/5 model wins | 181% avg, 5/5 model wins | Official Linux Vulkan target; all five catalog rows validate on the public Intel node, with performance tuning still younger than RDNA4 |
| Apple Silicon / Metal | 5 | 87% avg, 1 model win | 54% avg, 1 model win | Mixed by model; Gemma 31B and Qwen 35B are closest |

Full per-model numbers are in [Benchmarks](#benchmarks) and on the public
dashboard: [zolotukhin.ai/zinc/benchmarks](https://zolotukhin.ai/zinc/benchmarks).

## Start Here

### Install a prebuilt binary

Prebuilt binaries are installed from GitHub Releases. Until the first release is
published, build from source below. After a release exists, one command works on
Linux x86_64 (Vulkan) or Apple Silicon macOS (Metal):

```bash
curl -fsSL https://raw.githubusercontent.com/zolotukhin/zinc/main/scripts/install.sh | bash
```

The installer downloads the latest published release for your platform, verifies
its SHA-256 checksum, installs under `~/.local/share/zinc`, and links the binary
into `~/.local/bin/zinc`. Pin a version with `ZINC_VERSION=vX.Y.Z`. Prefer
manual? Grab a tarball and `SHA256SUMS.txt` from the
[releases page](https://github.com/zolotukhin/zinc/releases), or read
[`scripts/install.sh`](scripts/install.sh) before piping it into `bash`.

### Or build from source

Works the same on Linux (AMD or Intel GPU) and macOS (Apple Silicon):

```bash
git clone https://github.com/zolotukhin/zinc.git
cd zinc
zig build -Doptimize=ReleaseFast

# On RDNA4 Linux, enable cooperative matrix.
# Skip this on Intel Arc and macOS.
export RADV_PERFTEST=coop_matrix

# Verify GPU, shaders, and runtime
./zig-out/bin/zinc --check

# See which models fit this machine
./zig-out/bin/zinc model list

# Download a model
./zig-out/bin/zinc model pull qwen35-9b-q4k-m

# Run a prompt (--chat applies the model's chat template for instruct models)
./zig-out/bin/zinc --model-id qwen35-9b-q4k-m --prompt "Hello" --chat

# Or download any GGUF repo straight from Hugging Face (llama.cpp-style -hf)
./zig-out/bin/zinc -hf Qwen/Qwen3-0.6B-GGUF:Q8_0 --prompt "Hello" --chat

# Or open the chat UI in your browser
./zig-out/bin/zinc chat
```

The server exposes the built-in chat UI at `/` and an OpenAI-compatible API at `/v1`.

## What Works Today

ZINC is usable today as a local, single-user inference engine for the
validated models listed below.

| Area | What you can do today |
|------|------------------------|
| Run models | Use the CLI for single-stream inference on supported GGUF models |
| Chat | Start the built-in browser UI with `zinc chat`, including streaming and thinking-mode display |
| API | Serve OpenAI-compatible `/v1` endpoints with streaming responses |
| Models | Manage catalog models with `list`, `pull`, `use`, `active`, and `rm` |
| AMD GPUs | Run the Vulkan backend with RDNA-tuned wave64, cooperative-matrix, and fused-op shaders |
| Intel Arc | Run the Linux Vulkan backend on Arc Xe2/Battlemage GPUs with the same managed catalog and benchmark harness |
| Apple Silicon | Run the native Metal backend with MSL shaders, zero-copy mmap, and simdgroup ops |
| Setup | Let ZINC select the available backend at build time: Vulkan on Linux, Metal on macOS |

## Still Rough

- Continuous batching and multi-tenant serving are still roadmap work
- The supported-model list is intentionally narrow
- Apple Silicon and Intel Arc performance tuning are ongoing (RDNA4 path is more mature)

## The Problem

Consumer GPUs have the hardware for fast LLM inference — bandwidth, compute, VRAM — but the software doesn't use it:

- **AMD RDNA3/RDNA4**: ROCm doesn't support them. vLLM requires ROCm. llama.cpp's Vulkan path has no RDNA-specific tuning. These $500–1500 cards sit idle.
- **Intel Arc**: Arc B-series has the VRAM and Vulkan support to run useful local models, but most local inference stacks treat it as a compatibility path rather than a first-class target.
- **Apple Silicon**: MLX and llama.cpp Metal work, but leave performance on the table. No engine is built from scratch around Metal's strengths (unified memory, simdgroup ops, zero-copy mmap).

## The Solution

ZINC builds an inference engine tuned for the hardware you actually have.

**Hand-tuned shaders for each platform.** On AMD: wave64, cooperative matrix, architecture-aware tiling via Vulkan compute. On Intel Arc: the same Linux Vulkan runtime with Arc-aware device detection, model-fit checks, and benchmark coverage. On Apple Silicon: native MSL kernels with simdgroup reductions, zero-copy model loading, and Metal pipeline tuning. Not a generic backend that happens to run — built to extract real performance from each GPU.

**One binary, no driver stack.** No ROCm, no Python. Build with Zig, point at a GGUF, run inference. The right backend (Vulkan or Metal) is selected automatically at build time.

**Drop-in compatible.** OpenAI-compatible API, built-in chat UI, managed model catalog. Point your existing client at it and it works.

## Supported Models

The list below matches the current managed model catalog, not a broader wishlist.

- [Qwen 3.5 9B Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) — supported on AMD RDNA4 16/32 GB, Intel Arc, and Apple Silicon
- [Qwen3.6 35B-A3B UD Q4_K_XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) — supported on AMD RDNA4 32 GB, Intel Arc 32 GB, and Apple Silicon
- [Qwen3.6 27B Dense Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) — experimental on AMD RDNA4 32 GB, Intel Arc 32 GB, and Apple Silicon
- [Gemma 4 31B Q4_K_M](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) — supported on AMD RDNA4 32 GB, Intel Arc 32 GB, and Apple Silicon
- [Gemma 4 26B-A4B MoE Q4_K_M](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) — supported on AMD RDNA4 32 GB, Intel Arc 32 GB, and Apple Silicon

- Use `zinc model list --json` for machine-readable model metadata
- Current throughput and latency numbers live on the public benchmarks page: [zolotukhin.ai/zinc/benchmarks](https://zolotukhin.ai/zinc/benchmarks)

**Quantization formats**: Q4_K, Q5_K, Q6_K, Q8_0, Q5_0, MXFP4, F16, F32

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
./zig-out/bin/zinc --check --model-id qwen36-35b-a3b-q4k-xl
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

The README keeps the supported-model section concise and leaves the full managed-model workflow to the docs.

Use these for model selection, cache management, and API details:

- [Running ZINC](https://zolotukhin.ai/zinc/docs/running-zinc)
- [Serving HTTP API](https://zolotukhin.ai/zinc/docs/api)

### Run a Prompt

```bash
./zig-out/bin/zinc -m /path/to/model.gguf --prompt "The capital of France is"
```

Or skip the manual download and pass a Hugging Face repo (with an optional
`:quant` tag) — ZINC downloads the GGUF into its model cache on first use and
reuses it afterwards:

```bash
./zig-out/bin/zinc -hf Qwen/Qwen3-0.6B-GGUF:Q8_0 --prompt "The capital of France is"
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

The tables below are pulled directly from the published benchmark data at [zolotukhin.ai/zinc/benchmarks](https://zolotukhin.ai/zinc/benchmarks). Latest refreshes: RDNA 2026-07-01 UTC, Intel Arc 2026-07-07 UTC, Metal 2026-06-13 UTC. Numbers are median tok/s across the suite's runs, with ZINC and llama.cpp on the same hardware, weights, and prompt.

### AMD RDNA4 — Radeon AI PRO R9700 (Vulkan)

| Model | ZINC prefill | llama.cpp prefill | ZINC % | ZINC decode | llama.cpp decode | ZINC % |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.6 35B A3B UD Q4_K_XL | **540.33** | 397.08 | **136%** | **166.80** | 108.54 | **154%** |
| Qwen 3.5 9B Q4_K_M | **738.97** | 549.04 | **135%** | **97.46** | 85.47 | **114%** |
| Qwen 3.6 27B Dense Q4_K_M | **212.79** | 183.76 | **116%** | **31.97** | 30.65 | **104%** |
| Gemma 4 26B-A4B MoE Q4_K_M | **809.16** | 496.83 | **163%** | **113.74** | 102.08 | **111%** |
| Gemma 4 31B Q4_K_M | **248.58** | 199.58 | **125%** | **28.81** | 28.54 | **101%** |

### Apple Silicon M4 Max (Metal)

| Model | ZINC prefill | llama.cpp prefill | ZINC % | ZINC decode | llama.cpp decode | ZINC % |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 26B-A4B MoE Q4_K_M | 327.87 | 407.46 | 81% | 69.51 | 82.81 | 83% |
| Gemma 4 31B Q4_K_M | **131.07** | 102.28 | **128%** | 22.68 | 22.70 | 100% |
| Qwen 3.5 9B Q4_K_M | 36.53 | 332.65 | 11% | 29.42 | 57.87 | 52% |
| Qwen 3.6 27B Dense Q4_K_M | 15.87 | 104.34 | 15% | 15.44 | 21.93 | 70% |
| Qwen 3.6 35B A3B UD Q4_K_XL | 97.17 | 300.71 | 33% | **81.64** | 63.09 | **131%** |

### Intel Arc — Intel(R) Graphics BMG G31 (Vulkan)

| Model | ZINC prefill | llama.cpp prefill | ZINC % | ZINC decode | llama.cpp decode | ZINC % |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.6 35B A3B UD Q4_K_XL | **191.18** | 135.39 | **141%** | **75.26** | 75.07 | **100%** |
| Qwen 3.5 9B Q4_K_M | **191.06** | 141.98 | **135%** | **55.98** | 54.00 | **104%** |
| Qwen 3.6 27B Dense Q4_K_M | **92.21** | 37.07 | **249%** | **20.01** | 19.23 | **104%** |
| Gemma 4 26B-A4B MoE Q4_K_M | **492.97** | 247.57 | **199%** | **64.98** | 62.43 | **104%** |
| Gemma 4 31B Q4_K_M | **120.83** | 67.23 | **180%** | **18.01** | 17.37 | **104%** |

### Where we stand vs llama.cpp

- **Ahead of llama.cpp on RDNA4**: aggregate prefill and decode are ahead for all five published RDNA models in the latest suite. Qwen 3.6 35B-A3B decode is `1.54x`, Gemma 4 26B MoE decode is `1.11x`, and Gemma 4 31B dense decode is narrowly ahead at `1.01x`.
- **Intel Arc is official now**: the current Intel Arc Vulkan matrix completes all five catalog rows, with ZINC ahead on both prefill and decode for every headline model. The margins are smaller than RDNA4 on decode and the path is still younger operationally, but it is now a supported target.
- **Still close**: Gemma 4 31B long-context decode remains a tight row even though the model-level RDNA result is ahead overall.
- **Metal is mixed by model**: Gemma 4 31B prefill and Qwen 3.6 35B decode are ahead of llama.cpp, Gemma 4 31B decode is essentially tied, and the smaller Qwen dense rows still need backend-specific tuning.

For local benchmark commands, harnesses, and methodology, see:

- [Development Guide](./docs/DEVELOPMENT.md)
- [Running ZINC](./docs/RUNNING_ZINC.md)

## Current Status

| Component | Status |
|-----------|--------|
| Vulkan infrastructure | Done |
| GGUF parser + model loader | Done |
| GPU detection (AMD/Intel Vulkan) | Done |
| Native BPE tokenizer (from GGUF) | Done |
| GLSL compute shaders (16) | Done |
| Compute graph + architecture builders | Done |
| Forward pass (decode loop) | Working — 166.80 tok/s on RDNA4, 75.26 tok/s on Intel Arc, and 81.64 tok/s on Apple M4 Max for Qwen 3.6 35B-A3B |
| Forward pass (prefill loop) | Working — 540.33 tok/s on RDNA4 and 191.18 tok/s on Intel Arc for Qwen 3.6 35B-A3B; Metal prefill is fast on Qwen 3 8B and Gemma 4 31B but uneven across the catalog |
| GPU SSM shaders + cmd batching | Done — RDNA decode is 166.80 tok/s on Qwen 3.6 35B |
| HTTP server + OpenAI API | Done — Qwen 35B-A3B raw API ~100 tok/s on RDNA4 and Metal server path in progress |
| Continuous batching | Phase 4 |
| TurboQuant KV compression | Phase 5 |

Validated on AMD Radeon AI PRO R9700 (RDNA4) and Intel Arc BMG G31-class hardware: Vulkan 1.3 init, GGUF parsing, large catalog models loaded to VRAM, MoE graphs built, coherent inference output verified, and public benchmark rows published against llama.cpp on the same machines.

## Next Steps

The next push is turning the clean RDNA headline sweep into a wider, harder-to-move advantage:

1. **Widen the Gemma 4 31B margin** — it is ahead in the current RDNA suite, but decode is only `1.01x` llama.cpp. This is the row most likely to regress if we stop paying attention.
2. **Extend the sweep across more scenarios** — the headline five-model board is green; long-context and long-draft cells need the same level of repeatable coverage.
3. **Keep improving MoE and SSM prefill** — RDNA prefill now wins the published rows, but the best kernel work is still in batching SSM projections, reducing recurrent-state traffic, and opening more hybrid fast paths.
4. **Harden Intel Arc and Metal** — Intel Arc is now in the official support matrix, and Apple Silicon remains public in the same benchmark format; both need the same repeatable tuning discipline RDNA already has.
5. **Keep the dashboard honest** — all public claims should come from the fair server-vs-server harness, not one-off CLI runs or mixed warmup conditions.

The detailed cycle-50 field report is in the [RDNA optimization blog post](https://zolotukhin.ai/blog/2026-04-26-the-gate-that-keeps-qwen-35b-prefill-at-half-of-llama-cpp-on-rdna4).

## License

MIT
