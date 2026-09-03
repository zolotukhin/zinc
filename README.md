<p align="center">
  <img src="assets/zinc_trademark_new.png" alt="ZINC" width="360">
</p>

# ZINC

ZINC is a local LLM inference engine written in Zig. It runs GGUF models from a
command line, a browser chat UI, or an OpenAI-compatible API.

The project is built around consumer GPUs:

- AMD Radeon through Vulkan or ROCm
- Intel Arc through Vulkan
- Apple Silicon through Metal
- NVIDIA RTX through an experimental CUDA backend

There is no Python service to keep alive. The model loader, tokenizer, GPU
runtime, server, chat UI, and model manager ship together.

<p>
  <a href="https://github.com/zolotukhin/zinc/actions/workflows/test.yml"><img src="https://github.com/zolotukhin/zinc/actions/workflows/test.yml/badge.svg" alt="CI status"></a>
  <a href="https://zolotukhin.ai/zinc"><img src="https://img.shields.io/badge/docs-zolotukhin.ai%2Fzinc-d35400" alt="ZINC documentation"></a>
  <a href="https://discord.gg/QRUgWH2aGV"><img src="https://img.shields.io/badge/Discord-Join%20ZINC-5865F2?logo=discord&logoColor=white" alt="ZINC Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a>
</p>

[Getting started](https://zolotukhin.ai/zinc/docs/getting-started) ·
[Benchmarks](https://zolotukhin.ai/zinc/benchmarks) ·
[API guide](https://zolotukhin.ai/zinc/docs/api) ·
[Hardware guide](https://zolotukhin.ai/zinc/docs/hardware-requirements) ·
[ROCm guide](docs/ROCM.md)

## A current result

On a Radeon AI PRO R9700, the checked-in ROCm suite completes six models. ZINC
is ahead of the llama.cpp build used for the run on prefill, decode, and the
combined score in every core row.

| Model | ZINC prefill | llama.cpp prefill | ZINC decode | llama.cpp decode | Overall |
|---|---:|---:|---:|---:|---:|
| Qwen 3.8 27B Q4_K_M | **391.76** | 150.51 | **32.27** | 29.83 | **122.5%** |
| Muse Glimmer 30B Q4_K_M | **629.33** | 386.32 | **30.15** | 27.90 | **110.4%** |
| Qwen 3.6 35B A3B Q4_K_XL | **581.18** | 506.12 | **80.89** | 66.34 | **121.4%** |
| Qwen 3.5 9B Q4_K_M | **1135.87** | 779.92 | **78.76** | 69.99 | **113.3%** |
| Gemma 4 26B-A4B Q4_K_M | **1166.49** | 623.75 | **80.28** | 70.08 | **117.3%** |
| Gemma 4 31B Q4_K_M | **423.03** | 189.39 | **24.95** | 23.06 | **111.8%** |

These are median tokens per second from reusable servers on the same GPU, model
file, prompts, and run counts. They are not cherry-picked single runs. The
[benchmark dashboard](https://zolotukhin.ai/zinc/benchmarks#rdna-rocm) has the
four workloads, sample spreads, build revisions, and results for Vulkan, Metal,
CUDA, and Intel Arc as well.

## Get running

ZINC currently expects Zig 0.15.2 or newer. Linux builds also need a Vulkan
loader and `glslc`; ROCm builds need a working ROCm installation. macOS uses the
native Metal backend by default.

```bash
git clone https://github.com/zolotukhin/zinc.git
cd zinc
zig build -Doptimize=ReleaseFast

# Check the GPU, runtime, shaders, and model fit before the first run.
./zig-out/bin/zinc --check

# Download a small managed model and run it.
./zig-out/bin/zinc model pull qwen35-9b-q4k-m
./zig-out/bin/zinc --model-id qwen35-9b-q4k-m --prompt "Hello" --chat
```

On RDNA4 with the Vulkan backend, enable cooperative matrices before checking
or benchmarking:

```bash
export RADV_PERFTEST=coop_matrix
./zig-out/bin/zinc --check
```

To build the ROCm backend instead:

```bash
ROCM_PATH=/opt/rocm zig build -Dbackend=rocm -Doptimize=ReleaseFast
ROCR_VISIBLE_DEVICES=0 ./zig-out/bin/zinc --check
```

`ReleaseFast` matters for real use and performance measurements. A plain debug
build is useful for development, but it is not a fair throughput baseline.

Prebuilt binaries will be available through GitHub Releases. Until the first
release is published, build from source as shown above.

## Run a model

Use a managed catalog entry:

```bash
./zig-out/bin/zinc model list
./zig-out/bin/zinc model pull qwen35-9b-q4k-m
./zig-out/bin/zinc --model-id qwen35-9b-q4k-m --prompt "Explain wavefront occupancy" --chat
```

Or point ZINC at a GGUF file or Hugging Face repository:

```bash
./zig-out/bin/zinc -m /path/to/model.gguf --prompt "The capital of France is"
./zig-out/bin/zinc -hf Qwen/Qwen3-0.6B-GGUF:Q8_0 --prompt "Hello" --chat
```

Start the local server and browser UI with:

```bash
./zig-out/bin/zinc chat --model-id qwen35-9b-q4k-m
```

The chat UI is served at `/`, health checks at `/health`, and the API at `/v1`.
Existing clients can use `POST /v1/chat/completions`, including streaming and
tool-call responses. See the [API guide](https://zolotukhin.ai/zinc/docs/api)
for request examples and SDK configuration.

## Models in current work

This is a working set, not a promise that every backend has identical coverage.
The runtime column makes the distinction explicit without splitting the models
into separate lists.

| Model | Runtime | Current use |
|---|---|---|
| [Qwen 3.5 9B Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | Local GGUF | Broad hardware smoke target and compact daily model |
| [Qwen 3.6 35B-A3B Q4_K_XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | Local GGUF | Hybrid/MoE tuning across AMD, Intel, and Apple GPUs |
| [Qwen 3.8 27B Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | Local GGUF | Current ROCm decode and NextN drafting target |
| Muse Glimmer 30B Q4_K_M | Local GGUF | ROCm and Metal dense-model work |
| [Muse Spark 1.3](https://research.meta.ai/blog/introducing-muse-spark-1-3) | Hosted model | Coding and agent-workflow reference alongside the local work |
| [Gemma 4 26B-A4B Q4_K_M](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | Local GGUF | Sparse MoE and sliding-window attention coverage |
| [Gemma 4 31B Q4_K_M](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) | Local GGUF | Dense Gemma coverage |

Muse Spark is included because the project site follows the broader model and
agent landscape. It is accessed through Meta's hosted service and is not a ZINC
GGUF compatibility claim.

Current local quantization support includes Q4_K, Q5_K, Q6_K, Q8_0, Q5_0,
MXFP4, F16, and F32. Run `zinc model list --json` for machine-readable catalog
metadata and fit estimates.

## Platform status

| Platform | Backend | Status |
|---|---|---|
| AMD RDNA4 Linux | Vulkan | Supported and tuned |
| AMD RDNA4 Linux | ROCm/HIP | Supported; current reference target is `gfx1201` |
| AMD RDNA3 Linux | Vulkan | Supported |
| Intel Arc Xe2 / Battlemage Linux | Vulkan | Supported |
| Apple Silicon | Metal | Supported with native MSL kernels |
| NVIDIA RTX | CUDA | Experimental |

AMD users choose the backend at build time. Vulkan is the broad path; ROCm has
its own HIP runtime and kernels. Benchmark results for the two are kept separate
because the driver stacks and builds are different.

## What works today

- Single-stream GGUF inference from the CLI
- A browser chat interface with streaming and thinking-mode display
- OpenAI-compatible chat completions and model listing
- Managed model download, selection, removal, and fit checks
- Native GPU paths for Vulkan, ROCm, and Metal
- Automatic Qwen 3.8 NextN drafting on ROCm when the GGUF contains the draft block
- Reproducible same-machine benchmark tooling against llama.cpp

The project is still active engineering work. Continuous batching and
multi-tenant serving are not complete. Gemma 26B MoE on ROCm currently uses one
request slot, and performance is still uneven by model on Metal and CUDA. The
dashboard keeps incomplete and failed rows visible instead of averaging them
away.

## Development

```bash
zig build -Doptimize=ReleaseFast
zig build test
./zig-out/bin/zinc --check
```

The repository also has Bun tests for the browser and API surfaces. The
[development guide](docs/DEVELOPMENT.md) covers the full test suite, benchmark
harnesses, graph export, and debugging workflow.

Useful references:

- [Running ZINC](docs/RUNNING_ZINC.md)
- [Hardware requirements](docs/HARDWARE_REQUIREMENTS.md)
- [ROCm backend](docs/ROCM.md)
- [Contributing](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

## Architecture

ZINC keeps the frontend shared and the GPU work backend-specific. GGUF loading,
tokenization, graph construction, sampling, model management, and the HTTP API
feed native Vulkan, ROCm, Metal, or CUDA execution paths.

<p align="center">
  <img src="assets/architecture.svg" alt="ZINC architecture" width="680">
</p>

## License

MIT
