# Development Guide

Everything you need to build, test, debug, and contribute to ZINC.

If you just want to run inference, see [Getting Started](/zinc/docs/getting-started/). This page is for people who want to modify the engine, add features, fix bugs, or understand the internals.

## Prerequisites

| Tool                | Version            | What it does                                                                  |
| ------------------- | ------------------ | ----------------------------------------------------------------------------- |
| **Zig**             | 0.17.0             | Host compiler for all Zig source                                              |
| **Bun**             | 1.4.4 minimum      | TypeScript tests, package management, site scripts, and `bun fmt`             |
| **glslc**           | Current LunarG SDK | Shader compiler with `GL_EXT_integer_dot_product` support (Linux Vulkan only) |
| **Vulkan SDK**      | 1.3+               | Runtime for AMD and Intel Arc GPU dispatch (Linux only)                       |
| **Xcode CLI Tools** | Latest             | Metal compiler and frameworks (macOS only)                                    |

On macOS (Apple Silicon):

```bash
brew install bun
xcode-select --install
```

On Linux (AMD or Intel Arc GPU):

```bash
# Zig 0.17.0 is selected and downloaded by scripts/zig.sh when needed.
# Bun: https://bun.sh
# Install glslc from the current LunarG Vulkan SDK.
sudo apt install libvulkan-dev
```

## Build

```bash
git clone https://github.com/zolotukhin/zinc.git
cd zinc

# Debug build (fast compile, slow runtime)
./scripts/zig.sh build

# Release build (slow compile, fast runtime — use for benchmarking)
./scripts/zig.sh build -Doptimize=ReleaseFast
```

The binary lands at `./zig-out/bin/zinc`. Shaders are compiled from `src/shaders/*.comp` to SPIR-V during the build.

## Test

Use Bun 1.4.4 or newer. `.bun-version` pins the version used by CI. Install
the client tooling with the committed text lockfile:

```bash
bun install --frozen-lockfile
```

```bash
# CPU-only Zig and Bun tests (default)
./scripts/zig.sh build test

# Just the CPU-only Bun tests
bun test loops/ tools/ site/src/ tests/chat_ui_markdown.test.ts

# On a dedicated Metal worker: include hardware unit tests without model smoke tests
./scripts/zig.sh build test -Dgpu-tests=true

# On a dedicated GPU worker: include model and live API smoke tests
ZINC_API_BASE_URL=http://localhost:8080/v1 ./scripts/zig.sh build test -Dfull-tests=true
```

The default Metal suite reports hardware cases as skipped before they create
a device, allocate GPU buffers, or dispatch a kernel. Its test-only shim also
refuses device creation, so an unguarded test cannot silently use the GPU.
`-Dgpu-tests=true` enables those cases on a dedicated worker; `-Dfull-tests=true`
enables them together with the model/API smoke tests. Hosted macOS virtual GPUs
do not qualify as supported Apple Silicon hardware.

The test suite covers:

- Zig unit tests across all modules (tokenizer, routes, catalog, forward pass, etc.)
- Chat UI rendering, markdown, thinking logic, repetition detection
- API benchmark tool validation
- Site build and Zig API documentation generation
- Optional: OpenAI SDK compatibility, Qwen model smoke tests

## Format

```bash
bun run format
bun run format:check
```

These commands use `bun fmt` on supported files changed from `origin/main`.
Pass paths directly to format a particular file, for example
`bun fmt tools/benchmark_api.mjs`. CI checks files changed from the push or
pull-request base. Settings live in `bunfig.toml`; generated measurements and
byte-sensitive fixtures are excluded. Astro templates are checked by the site
build and browser validation because Bun's native formatter does not support
`.astro` without a Prettier fallback. Do not add Prettier to format them.

The website also needs the Node version pinned in `site/.node-version`.
From `site/`, run `bun install --frozen-lockfile` and `bun run build`.

## Debug Flags

| Flag                        | What it does                                                      |
| --------------------------- | ----------------------------------------------------------------- |
| `--debug`                   | Enable verbose logging (same as `ZINC_DEBUG=1`)                   |
| `--profile`                 | Enable per-dispatch GPU profiling (Vulkan only)                   |
| `ZINC_DEBUG=1`              | Environment variable alternative to `--debug`                     |
| `ZINC_PREFILL_PROFILE=1`    | Per-phase prefill timing (dense FFN, SSM, attention, projections) |
| `ZINC_BATCHED_PREFILL=0`    | Disable RDNA batched prefill; useful for isolating regressions    |
| `ZINC_FA_SPLIT_K=<n>`       | Override the scalar flash-attention split-K count                 |
| `ZINC_TOOL_CALLING=0`       | Disable OpenAI-compatible tool calling on `/v1/chat/completions`  |
| `RADV_PERFTEST=coop_matrix` | Enable cooperative matrix on RDNA4 (recommended)                  |
| `RADV_DEBUG=shaders`        | Dump compiled Vulkan shaders                                      |
| `RADV_DEBUG=shaderstats`    | Show VGPR/occupancy/spill stats                                   |

For the full set of `ZINC_*` knobs (fusion gates, per-architecture experiments), grep the codebase: `rg 'getEnvVar.*ZINC_' src/`.

Example debug run:

```bash
ZINC_DEBUG=1 ./zig-out/bin/zinc -m model.gguf --prompt "Hello" -n 64
```

## Find the code

Use the [codebase map](/zinc/docs/codebase-map/) for source ownership, backend
entry points, tests, standalone tools, and release automation. CLI `--help`
and implementation source are authoritative for current flags.

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

Run model inference and benchmarks on a dedicated GPU worker. Check existing
GPU users first and stop only processes owned by your task. The project
workstation is not an unattended inference worker.

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

# Serving concurrency gate (expected to fail until continuous batching lands)
bun tools/benchmark_api.mjs \
  --base http://localhost:8080 \
  --mode concurrency \
  --min-c4-aggregate-scale 2.5 \
  --max-c4-p95-latency-multiplier 2.0
```

The concurrency gate compares each `concurrency=4` scenario against its
matching `concurrency=1` baseline. A serialized server typically shows
aggregate completion throughput near `1x` and p95 latency near `4x`; continuous
batching should raise aggregate throughput materially without linear p95 growth.

### Public performance output

```bash
# Generate the benchmark artifact that powers /zinc/benchmarks.
# RDNA runs over SSH; the local process only drives the remote worker.
bun tools/performance_suite.mjs --target rdna --rdna-sync --rdna-build --rdna-start-llama --output /tmp/zinc-performance.json
bun tools/performance_suite.mjs --target intel --output /tmp/zinc-intel-performance.json
```

The generated target entries include the exact ZINC git version/commit for the machine that produced each target and the llama.cpp binary version/commit used for the baseline on that target. The public performance page renders the same provenance so benchmark rows are tied to a concrete source tree and baseline build.

The Metal target runs on the machine that invokes the suite. Use a dedicated
Mac with explicit permission for that GPU run. `both` includes Metal; `all`
also includes the Intel and CUDA targets.

Use the RDNA suite's server-vs-server path for headline comparisons. One-shot CLI runs are diagnostics; do not compare them against a warmed baseline server when deciding whether a target is beaten.

### Hot decode kernel microbenchmarks

```bash
# Measure individual GPU kernel performance
./scripts/zig.sh build hot-bench -Doptimize=ReleaseFast
./zig-out/bin/zinc-hot-bench --shader-dir zig-out/share/zinc/shaders
```

For detailed tuning guidance, see [RDNA4 Tuning Guide](/zinc/docs/rdna4-tuning/), the [AMD GPU Reference](/zinc/docs/amd-gpu-reference/), and the [Intel GPU Reference](/zinc/docs/intel-gpu-reference/).

## Remote Benchmark Nodes

For AMD GPU testing, the project uses remote RDNA4 nodes. Intel Arc benchmark nodes use a separate environment namespace so published support runs do not collide with RDNA credentials or workdirs.

### RDNA4 nodes

Environment setup:

```bash
# .env file in repo root
# Optional selector for machines with more than one RDNA node.
# The selector maps rdna1 -> ZINC_RDNA1_* and rdna2 -> ZINC_RDNA2_*.
ZINC_RDNA_NODE=rdna1

ZINC_RDNA1_HOST=<ip-or-hostname>
ZINC_RDNA1_PORT=<ssh-port>
ZINC_RDNA1_USER=root

ZINC_RDNA2_HOST=<ip-or-hostname>
ZINC_RDNA2_PORT=<ssh-port>
ZINC_RDNA2_USER=root

# Compatibility fallback for older scripts and ad-hoc shell commands.
ZINC_HOST=<active-rdna-ip-or-hostname>
ZINC_PORT=<active-rdna-ssh-port>
ZINC_USER=<active-rdna-ssh-user>

# Optional overrides consumed by scripts/deploy_rdna4_server.sh
ZINC_REMOTE_DIR=/root/zinc                 # remote checkout path
ZINC_REMOTE_MODEL=/root/models/<file>.gguf # model file the deployed server loads
ZINC_SERVER_PORT=9090                      # remote server port
ZINC_REMOTE_LOG=/tmp/zinc_<port>.log       # remote log file
```

Node-specific deploy overrides can use the same prefix, for example
`ZINC_RDNA2_REMOTE_MODEL`, `ZINC_RDNA2_SERVER_PORT`, and
`ZINC_RDNA2_REMOTE_LOG`.

Intel Arc benchmark nodes use separate keys so they do not override the RDNA defaults:

```bash
ZINC_INTEL_HOST=<ip>
ZINC_INTEL_PORT=<ssh-port>
ZINC_INTEL_USER=<ssh-user>
ZINC_INTEL_WORKDIR=/home/<ssh-user>/zinc-gpu-loop
ZINC_INTEL_XDG_CACHE_HOME=/home/<ssh-user>/.cache
ZINC_INTEL_MODEL_ROOT=/home/<ssh-user>/.cache/zinc/models/models
# Temporary password-only nodes only; prefer key-based SSH after bootstrap.
ZINC_INTEL_SSH_PASSWORD_FILE=/tmp/zinc-intel-password
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
- **Model architectures** — Qwen3, Qwen3.5, and Gemma 4 (dense transformer, MoE, and SSM hybrid)
- **GGUF parsing** — zero-copy mmap with DMA to GPU VRAM
- **Vulkan init** — single-device, single-queue, push-constant dispatch
- **Metal init** — default system device, zero-copy `newBufferWithBytesNoCopy`

## Contributing

See [CONTRIBUTING.md](https://github.com/zolotukhin/zinc/blob/main/CONTRIBUTING.md) for development expectations, workflow, and what makes a good contribution.

See [Code of Conduct](https://github.com/zolotukhin/zinc/blob/main/CODE_OF_CONDUCT.md) for community standards.

## Further Reading

- [Getting Started](/zinc/docs/getting-started/) — first run, model download, basic usage
- [Running ZINC](/zinc/docs/running-zinc/) — CLI reference, server mode, managed models
- [API Reference](/zinc/docs/api/) — OpenAI-compatible HTTP endpoints
- [Technical Specification](/zinc/docs/spec/) — architecture, kernels, scheduler
- [RDNA4 Tuning Guide](/zinc/docs/rdna4-tuning/) — performance profiling and optimization
- [AMD GPU Reference](/zinc/docs/amd-gpu-reference/) — RDNA3/RDNA4 hardware details
- [Intel GPU Reference](/zinc/docs/intel-gpu-reference/) — Arc B-series hardware, memory bandwidth, and Xe2 opcode notes
- [Apple Silicon Reference](/zinc/docs/apple-silicon-reference/) — M1–M5 capabilities
- [Apple Metal Reference](/zinc/docs/apple-metal-reference/) — MSL kernel optimization
