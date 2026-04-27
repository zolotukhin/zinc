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
  <a href="https://discord.gg/tNDEgTG5s">
    <img src="https://img.shields.io/badge/Discord-Join%20ZINC-5865F2?logo=discord&logoColor=white" alt="ZINC Discord">
  </a>
</p>

> Local LLM inference on AMD GPUs and Apple Silicon — no ROCm, no MLX, one binary.

<p align="center">
  <img src="assets/zinc-chat-demo.gif" alt="ZINC Chat Demo — streaming inference on AMD RDNA4" width="720">
  <br>
  <em>35B parameter model running locally — Zig + Vulkan/Metal, no ROCm, no MLX</em>
</p>

## Supported Platforms

| Platform | GPU | Backend | Status |
|----------|-----|---------|--------|
| **Linux** | AMD RDNA4 (RX 9070, AI PRO R9700) | Vulkan | Primary — hand-tuned shaders |
| **Linux** | AMD RDNA3 (RX 7900 XTX, etc.) | Vulkan | Supported |
| **macOS** | Apple Silicon (M1, M2, M3, M4, M5) | Metal | Supported — native MSL shaders |

## Start Here

Works the same on Linux (AMD GPU) and macOS (Apple Silicon):

```bash
git clone https://github.com/zolotukhin/zinc.git
cd zinc
zig build -Doptimize=ReleaseFast

# On RDNA4 Linux, enable cooperative matrix
export RADV_PERFTEST=coop_matrix  # skip on macOS

# Verify GPU, shaders, and runtime
./zig-out/bin/zinc --check

# See which models fit this machine
./zig-out/bin/zinc model list

# Download a model
./zig-out/bin/zinc model pull qwen3-8b-q4k-m

# Run a prompt (--chat applies the model's chat template for instruct models)
./zig-out/bin/zinc --model-id qwen3-8b-q4k-m --prompt "Hello" --chat

# Or open the chat UI in your browser
./zig-out/bin/zinc chat
```

The server exposes the built-in chat UI at `/` and an OpenAI-compatible API at `/v1`.

## What Works Today

- Single-stream CLI inference on the validated models listed below
- OpenAI-compatible `/v1` API with streaming
- Built-in browser chat UI with thinking mode support
- Managed model workflow: `list`, `pull`, `use`, `active`, `rm`
- `zinc chat` — start server and open browser in one command
- **AMD path**: RDNA4-tuned Vulkan shaders (wave64, cooperative matrix, fused ops)
- **Apple Silicon path**: native Metal shaders (MSL, zero-copy mmap, simdgroup ops)
- Auto-detection: ZINC picks the right backend (Vulkan or Metal) at build time

## Still Rough

- Continuous batching and multi-tenant serving are still roadmap work
- The supported-model list is intentionally narrow
- Apple Silicon performance tuning is ongoing (RDNA4 path is more mature)

## The Problem

Consumer GPUs have the hardware for fast LLM inference — bandwidth, compute, VRAM — but the software doesn't use it:

- **AMD RDNA3/RDNA4**: ROCm doesn't support them. vLLM requires ROCm. llama.cpp's Vulkan path has no RDNA-specific tuning. These $500–1500 cards sit idle.
- **Apple Silicon**: MLX and llama.cpp Metal work, but leave performance on the table. No engine is built from scratch around Metal's strengths (unified memory, simdgroup ops, zero-copy mmap).

## The Solution

ZINC builds an inference engine tuned for the hardware you actually have.

**Hand-tuned shaders for each platform.** On AMD: wave64, cooperative matrix, architecture-aware tiling via Vulkan compute. On Apple Silicon: native MSL kernels with simdgroup reductions, zero-copy model loading, and Metal pipeline tuning. Not a generic backend that happens to run — built to extract real performance from each GPU.

**One binary, no driver stack.** No ROCm, no CUDA, no Python. Build with Zig, point at a GGUF, run inference. The right backend (Vulkan or Metal) is selected automatically at build time.

**Drop-in compatible.** OpenAI-compatible API, built-in chat UI, managed model catalog. Point your existing client at it and it works.

## Supported Models

The list below matches the current managed model catalog, not a broader wishlist.

- [Qwen3.5 35B-A3B UD Q4_K_XL](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) — supported on AMD RDNA4 32 GB and Apple Silicon
- [Qwen3.6 35B-A3B UD Q4_K_XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) — experimental on AMD RDNA4 32 GB and Apple Silicon
- [OpenAI GPT-OSS 20B Q4_K_M](https://huggingface.co/bartowski/openai_gpt-oss-20b-GGUF) — supported on Apple Silicon
- [Qwen3 8B Q4_K_M](https://huggingface.co/unsloth/Qwen3-8B-GGUF) — supported on AMD RDNA4 32 GB and Apple Silicon
- [Gemma 4 31B Q4_K_M](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) — supported on AMD RDNA4 32 GB and Apple Silicon
- [Gemma 4 12B (26B-A4B MoE) Q4_K_M](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) — experimental on AMD RDNA4 32 GB and Apple Silicon

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

The README keeps the supported-model section concise and leaves the full managed-model workflow to the docs.

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

The tables below are pulled directly from the latest published artifact at [zolotukhin.ai/zinc/benchmarks](https://zolotukhin.ai/zinc/benchmarks). Latest refresh: 2026-04-25 (RDNA4), 2026-04-22 (Metal). Numbers are median tok/s across the suite's runs on a fresh boot, ZINC and llama.cpp on the same hardware, weights, and prompt.

### AMD RDNA4 — Radeon AI PRO R9700 (Vulkan)

| Model | ZINC prefill | llama.cpp prefill | ZINC % | ZINC decode | llama.cpp decode | ZINC % |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 3 8B (dense) | **140.87** | 86.01 | **164%** | 70.44 | 106.15 | 66% |
| Qwen 3.5 35B A3B (MoE+SSM) | 66.58 | 183.64 | 36% | 82.21 | 103.45 | 79% |
| Qwen 3.6 35B A3B (MoE+SSM) | 69.22 | 182.10 | 38% | 82.34 | 102.77 | 80% |
| Gemma 4 26B A4B (MoE) | 43.33 | 333.65 | 13% | 85.93 | 98.32 | 87% |
| Gemma 4 31B (dense) | 54.60 | 108.09 | 51% | **33.72** | 28.53 | **118%** |
| GPT-OSS 20B (MoE) | 91.00 | 237.23 | 38% | 88.10 | 152.48 | 58% |

### Apple Silicon M4 Max (Metal)

| Model | ZINC prefill | llama.cpp prefill | ZINC decode | llama.cpp decode | ZINC % decode |
|---|---:|---:|---:|---:|---:|
| Qwen 3.5 35B A3B (MoE+SSM) | 3.5 | 119.72 | 46.67 | 73.36 | 64% |
| Qwen 3.6 35B A3B (MoE+SSM) | 3.3 | 128.73 | 38.35 | 78.72 | 49% |
| Qwen 3 8B (dense) | 9.2 | 79.32 | 32.83 | 90.12 | 36% |
| Gemma 4 31B (dense) | 0.2 | 74.14 | 0.23 | 25.79 | 1% |
| GPT-OSS 20B (MoE) | 10.4 | 599.25 | 26.1 | 109.15 | 24% |

### Where we stand vs llama.cpp

- **Ahead of llama.cpp**: Qwen 3 8B prefill on RDNA4 (1.6x), Gemma 4 31B dense decode on RDNA4 (1.18x).
- **Within striking distance (75–90% of llama.cpp)**: Qwen 3.5/3.6 35B-A3B decode on RDNA4, Gemma 4 26B A4B decode on RDNA4, Gemma 4 31B prefill is half — decode beats.
- **Active gap**: Qwen 3.5/3.6 35B-A3B prefill on RDNA4 sits at ~37% of llama.cpp because the entire batched prefill path is gated off for any model with `n_experts > 0` or `ssm_d_inner > 0`. The wire-up that closes this is documented in the [cycle-50 field report](https://zolotukhin.ai/blog/2026-04-26-the-gate-that-keeps-qwen-35b-prefill-at-half-of-llama-cpp-on-rdna4).
- **In flight**: most Metal prefill paths route through a per-token loop that does not amortize weight reads across prompt tokens. The Qwen 35B-A3B Metal decode at 38–47 tok/s is the realistic decode signal; the 0.2–3.5 tok/s prefill row reflects that the batched Metal MoE prefill landed for Gemma but is not yet wired across the catalog.

For local benchmark commands, harnesses, and methodology, see:

- [Development Guide](./docs/DEVELOPMENT.md)
- [Running ZINC](./docs/RUNNING_ZINC.md)

## Current Status

| Component | Status |
|-----------|--------|
| Vulkan infrastructure | Done |
| GGUF parser + model loader | Done |
| GPU detection (RDNA3/4) | Done |
| Native BPE tokenizer (from GGUF) | Done |
| GLSL compute shaders (16) | Done |
| Compute graph + architecture builders | Done |
| Forward pass (decode loop) | Working — 82.21 tok/s on RDNA4 and 46.67 tok/s on Apple M4 Max for Qwen 3.5 35B-A3B |
| Forward pass (prefill loop) | Working — 90+ tok/s on RDNA4 long-context for Qwen 3.6 35B-A3B; Metal prefill in flight |
| GPU SSM shaders + cmd batching | Done — RDNA decode is 82+ tok/s on Qwen 3.5/3.6 35B and 88 tok/s on GPT-OSS 20B |
| HTTP server + OpenAI API | Done — Qwen 35B-A3B raw API ~80 tok/s on RDNA4 and ~46 tok/s on Apple M4 Max |
| Continuous batching | Phase 4 |
| TurboQuant KV compression | Phase 5 |

Validated on AMD Radeon AI PRO R9700 (RDNA4): Vulkan 1.3 init, GGUF parsing, 21 GB model loaded to VRAM, 723-node MoE graph built, coherent inference output verified against CPU reference.

## Next Steps

The next push is closing the prefill gap to llama.cpp on hybrid MoE-plus-SSM models:

1. **Wire `mul_mm_q4k` into SSM proj prefill** — the tiled Q4_K GEMM is in the tree but only routes the language-model head where N=1 wastes the BN tile. The SSM proj fires 4 DMMVs per layer per token; batching them across the prompt is the deferred cycle-40 refactor.
2. **Port the `gated_delta_net.cu` block-resident state pattern** — today every prompt token re-reads and re-writes the full 2 MB SSM state per layer. Loading state once per workgroup and walking all tokens inside the kernel collapses 18 GB of state DRAM traffic per prefill to 4 MB.
3. **Open `canUseBatchedPrefillRdna` for MoE+SSM hybrids** — the entire batched prefill body (`flash_attn_batched`, `rope_batched`, `dmmv_q4k_batch_kpar`) is gated off when `n_experts > 0` or `ssm_d_inner > 0`. Once items 1 and 2 land, dropping the gate activates Br-row attention batching on the same workload.
4. **Land the cycle-50 micro-restructure pattern on MoE inner loops** — wider threads-per-row plus halved per-thread register slabs lifted ssm_delta_net by 2.7%. The same shape change is untried on `dmmv_q4k_moe_kpar` and `dmmv_q4k_moe_fused_down_acc`.
5. **Ship batched Metal prefill across the catalog** — the Gemma path landed; Qwen 3.5/3.6 and GPT-OSS still route through the per-token Metal path that produces the 0.2–10 tok/s prefill numbers above.

The full plan and 50-cycle field report is in the [cycle-50 blog post](https://zolotukhin.ai/blog/2026-04-26-the-gate-that-keeps-qwen-35b-prefill-at-half-of-llama-cpp-on-rdna4).

## License

MIT
