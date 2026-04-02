# Development Guide

Everything you need to build, test, debug, and contribute to ZINC.

If you just want to run inference, see [Getting Started](GETTING_STARTED.md). This page is for people who want to modify the engine, add features, fix bugs, or understand the internals.

## Prerequisites

| Tool | Version | What it does |
|------|---------|-------------|
| **Zig** | 0.15.2+ | Host compiler for all Zig source |
| **Bun** | Latest | TypeScript test runner, site builder, API benchmarks |
| **glslc** | shaderc 2023.8 | Compiles GLSL compute shaders to SPIR-V (Linux/AMD only) |
| **Vulkan SDK** | 1.3+ | Runtime for AMD GPU dispatch (Linux only) |
| **Xcode CLI Tools** | Latest | Metal compiler and frameworks (macOS only) |

On macOS (Apple Silicon):

```bash
brew install zig bun
xcode-select --install
```

On Linux (AMD GPU):

```bash
# Zig: https://ziglang.org/download/
# Bun: https://bun.sh
sudo apt install libvulkan-dev glslc
```

## Build

```bash
git clone https://github.com/zolotukhin/zinc.git
cd zinc

# Debug build (fast compile, slow runtime)
zig build

# Release build (slow compile, fast runtime — use for benchmarking)
zig build -Doptimize=ReleaseFast
```

The binary lands at `./zig-out/bin/zinc`. Shaders are compiled from `src/shaders/*.comp` to SPIR-V during the build.

## Test

```bash
# Full test suite (Zig unit tests + Bun TypeScript tests)
zig build test

# Just the TypeScript tests
bun test

# With integration smoke tests (requires a running server + model files)
ZINC_API_BASE_URL=http://localhost:8080/v1 zig build test -Dfull-tests=true
```

The test suite covers:
- Zig unit tests across all modules (tokenizer, routes, catalog, forward pass, etc.)
- Chat UI rendering, markdown, thinking logic, repetition detection
- API benchmark tool validation
- Site build and Zig API documentation generation
- Optional: OpenAI SDK compatibility, Qwen model smoke tests

## Debug Flags

| Flag | What it does |
|------|-------------|
| `--debug` | Enable verbose logging (same as `ZINC_DEBUG=1`) |
| `--profile` | Enable per-dispatch GPU profiling (Vulkan only) |
| `ZINC_DEBUG=1` | Environment variable alternative to `--debug` |
| `RADV_PERFTEST=coop_matrix` | Enable cooperative matrix on RDNA4 (recommended) |
| `RADV_DEBUG=shaders` | Dump compiled Vulkan shaders |
| `RADV_DEBUG=shaderstats` | Show VGPR/occupancy/spill stats |

Example debug run:

```bash
ZINC_DEBUG=1 ./zig-out/bin/zinc -m model.gguf --prompt "Hello" -n 64
```

## Project Structure

```
src/
├── main.zig                 # CLI entry point, arg parsing, server startup
├── compute/
│   ├── forward.zig          # Vulkan decode loop (~2000 lines)
│   ├── forward_metal.zig    # Metal decode loop (~3500 lines)
│   ├── dmmv.zig             # DMMV dispatch (quantized matmul-vec)
│   └── elementwise.zig      # Fused elementwise ops (RMS norm, SwiGLU, etc.)
├── model/
│   ├── tokenizer.zig        # BPE tokenizer, chat templates
│   ├── catalog.zig          # Managed model catalog
│   └── gguf.zig             # GGUF file parser
├── server/
│   ├── routes.zig           # OpenAI-compatible API + chat completions
│   ├── chat.html            # Built-in chat UI (embedded at compile time)
│   └── model_manager.zig    # Hot model switching
├── vulkan/                  # Vulkan backend (device, pipeline, memory)
├── metal/                   # Metal backend (device, pipeline, shim.m)
├── shaders/
│   ├── *.comp               # GLSL compute shaders (Vulkan/SPIR-V)
│   └── metal/*.metal        # MSL compute shaders (Apple Silicon)
├── gpu/
│   └── interface.zig        # Backend abstraction (Vulkan vs Metal)
site/                        # zolotukhin.ai Astro site
docs/                        # Technical documentation (published to site)
tools/                       # API benchmark, standalone utilities
specs/                       # Feature specifications and plans
benchmarks/                  # GPU microbenchmarks
scripts/                     # Deployment scripts
tests/                       # TypeScript test files
loops/                       # Self-improving optimization loop
```

## Graph Export

ZINC can export the decode computation graph for debugging and visualization:

```bash
# JSON report with node/edge lists, op counts, and critical path
./zig-out/bin/zinc -m model.gguf --graph-report graph.json

# Graphviz DOT for visualization
./zig-out/bin/zinc -m model.gguf --graph-dot graph.dot
dot -Tsvg graph.dot -o graph.svg

# Inspect with jq
cat graph.json | jq '.summary'
cat graph.json | jq '.nodes[] | select(.op_type == "dmmv") | {name, quant, rows, cols}'
```

These flags are available via `--help-all`.

## Benchmarking

### CLI decode throughput

```bash
# Single-stream decode (the primary metric)
ZINC_DEBUG=1 ./zig-out/bin/zinc -m model.gguf --prompt "The capital of France is" -n 128
```

### API benchmarks

```bash
# Chat endpoint matrix (short/medium/long prompts, concurrency 1/2/4)
bun tools/benchmark_api.mjs --base http://localhost:8080 --mode chat

# Raw completions throughput
bun tools/benchmark_api.mjs --base http://localhost:8080 --mode raw
```

### Hot decode kernel microbenchmarks

```bash
# Measure individual GPU kernel performance
zig build hot-bench -Doptimize=ReleaseFast
./zig-out/bin/zinc-hot-bench --shader-dir zig-out/share/zinc/shaders
```

For detailed tuning guidance, see [RDNA4 Tuning Guide](RDNA4_TUNING.md) and the [GPU Reference](GPU_REFERENCE.md).

## RDNA4 Test Node

For AMD GPU testing, the project uses a remote RDNA4 node. Environment setup:

```bash
# .env file in repo root
ZINC_HOST=<ip>
ZINC_PORT=<ssh-port>
ZINC_USER=root
```

Deploy and test:

```bash
# Full deploy: sync → build → restart → health check
bash scripts/deploy_rdna4_server.sh

# Skip steps as needed
bash scripts/deploy_rdna4_server.sh --no-build --no-restart
```

The deploy script includes a retry health check (30 attempts, 1s apart) to handle model loading time.

## Key Architecture Decisions

Before making changes to these areas, understand the existing design:

- **Compute graph IR** — the decode graph is built from GGUF metadata, not hand-coded
- **Model architectures** — Qwen3.5 (dense + MoE + SSM hybrid) is the primary target
- **GGUF parsing** — zero-copy mmap with DMA to GPU VRAM
- **Vulkan init** — single-device, single-queue, push-constant dispatch
- **Metal init** — default system device, zero-copy `newBufferWithBytesNoCopy`

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development expectations, workflow, and what makes a good contribution.

See [Code of Conduct](../CODE_OF_CONDUCT.md) for community standards.

## Further Reading

- [Getting Started](GETTING_STARTED.md) — first run, model download, basic usage
- [Running ZINC](RUNNING_ZINC.md) — CLI reference, server mode, managed models
- [API Reference](API.md) — OpenAI-compatible HTTP endpoints
- [Technical Specification](SPEC.md) — architecture, kernels, scheduler
- [RDNA4 Tuning Guide](RDNA4_TUNING.md) — performance profiling and optimization
- [GPU Reference](GPU_REFERENCE.md) — RDNA3/RDNA4 hardware details
- [Apple Silicon Reference](APPLE_SILICON_REFERENCE.md) — M1–M5 capabilities
- [Apple Metal Reference](APPLE_METAL_REFERENCE.md) — MSL kernel optimization
