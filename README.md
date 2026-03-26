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
</p>

> Making AMD consumer GPUs actually usable for LLM inference.

## The Problem

AMD's RDNA3/RDNA4 GPUs (RX 9070, Radeon AI PRO R9700, etc.) have excellent memory bandwidth (576+ GB/s) and hardware features (cooperative matrix, integer dot product), but:

1. **ROCm doesn't support them** — only MI-series datacenter GPUs
2. **vLLM requires ROCm** — so it can't use these GPUs at all
3. **llama.cpp Vulkan works** but treats RDNA4 as an afterthought — no RDNA4-specific tuning, SPIR-V toolchain incompatibilities, no tensor parallelism
4. **No solution handles parallel requests well** on these GPUs for production use

These cards cost $500–1500 (vs $15,000+ for MI300X) and sit in millions of desktops doing nothing during inference.

## The Solution

A purpose-built LLM inference engine written in **Zig** + **Vulkan compute**, targeting AMD RDNA3/RDNA4 consumer GPUs. OpenAI-compatible API, optimized for throughput on parallel requests.

## Quick Start

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Zig | 0.15.2+ | [ziglang.org/download](https://ziglang.org/download/) |
| Vulkan SDK | 1.3+ | `apt install libvulkan-dev vulkan-tools` (Linux) or `brew install vulkan-loader vulkan-headers` (macOS) |
| glslc | shaderc 2023.8 | `apt install glslc` (Linux only, Ubuntu 24.04) |
| Bun | 1.0+ | `curl -fsSL https://bun.sh/install \| bash` |

**Important**: On Linux with RDNA4, newer glslc versions (v2026.2+) cause a 5x performance regression. Use the system package version only.

### Build

```bash
# Clone
git clone https://github.com/zolotukhin/zinc.git
cd zinc

# Build (macOS: shaders skipped; Linux: shaders compiled automatically)
zig build

# Force shader compilation on any platform
zig build -Dshaders=true
```

The binary is placed in `zig-out/bin/zinc`. Compiled SPIR-V shaders go to `zig-out/share/zinc/shaders/`.

### Run Inference (CLI mode)

```bash
./zig-out/bin/zinc -m /path/to/model.gguf --prompt "The capital of France is"
```

### Run as Server (Phase 4 — in progress)

```bash
./zig-out/bin/zinc -m /path/to/model.gguf -p 8080
```

### Export Decode Graph Artifacts

```bash
# Machine-readable structural report for custom tooling
./zig-out/bin/zinc -m /path/to/model.gguf --graph-report decode-graph.json

# Graphviz DOT for quick rendering/debugging
./zig-out/bin/zinc -m /path/to/model.gguf --graph-dot decode-graph.dot

# Both at once
./zig-out/bin/zinc -m /path/to/model.gguf \
  --graph-report decode-graph.json \
  --graph-dot decode-graph.dot
```

### Visualize the Decode Graph

The graph export is intended for debugging and performance work before adding a richer runtime profiler. The two outputs serve different purposes:

- `decode-graph.json`: machine-readable structure for scripts, dashboards, or a future custom viewer
- `decode-graph.dot`: quick visual rendering through Graphviz

Typical workflow:

```bash
# 1. Export both artifacts
./zig-out/bin/zinc -m /path/to/model.gguf \
  --graph-report decode-graph.json \
  --graph-dot decode-graph.dot

# 2. Render the DOT file to SVG
dot -Tsvg decode-graph.dot -o decode-graph.svg

# 3. Open the rendered graph
open decode-graph.svg   # macOS
# xdg-open decode-graph.svg   # Linux
```

Useful JSON inspection commands:

```bash
# Top-level structural summary
jq '{name, node_count, edge_count, max_depth, max_parallel_width, critical_path_node_count}' decode-graph.json

# Which op types dominate the graph?
jq '.op_counts' decode-graph.json

# Show the structural critical path
jq '.critical_path' decode-graph.json

# Inspect nodes that lie on the critical path
jq '.nodes[] | select(.is_on_critical_path)' decode-graph.json
```

How to read the output:

- `op_counts` shows which logical ZINC operations dominate the decode DAG, such as `dmmv`, `flash_attn`, `rope`, and `swiglu`
- `max_depth` and `critical_path_node_count` describe the longest dependency chain through the graph
- `max_parallel_width` shows the widest layer of structurally independent work
- nodes marked `is_on_critical_path` are the best first candidates when you want to reduce total decode latency
- the current export is structural, not timed: it tells you where parallelism and dependencies exist, not yet how long each node took on GPU

The DOT export highlights critical-path nodes in red so you can see the longest chain immediately. The JSON export is the better source for automated analysis or a future in-browser visualizer.

### CLI Reference

```
Usage: zinc [options]
  -m, --model <path>       Path to GGUF model file (required)
  -p, --port <port>        Server port (default: 8080)
  -d, --device <id>        Vulkan device index (default: 0)
  -c, --context <size>     Context length (default: 4096)
  --parallel <n>           Max concurrent requests (default: 4)
  --prompt <text>          Single prompt (CLI mode, no server)
  --kv-quant <bits>        TurboQuant KV cache bits: 0/2/3/4 (default: 0=off)
  --graph-report <path>    Write decode-graph JSON report from GGUF metadata
  --graph-dot <path>       Write decode-graph Graphviz DOT from GGUF metadata
  -h, --help               Show this help
```

The JSON report includes node/edge lists, op-type counts, per-node depth, root/leaf flags, and the structural critical path. The DOT export is intended for Graphviz or downstream visualization tools.

### Tests

```bash
# Zig unit tests (18 tests)
zig build test

# TypeScript loop tests (34 tests)
bun test loops/

# All tests
zig build test && bun test loops/
```

## Self-Improving Optimization Loop

ZINC includes an AI-powered self-improving loop that iteratively builds, deploys, and fixes/optimizes the engine on real RDNA4 hardware.

### Setup

Create a `.env` file with your remote RDNA4 node credentials:

```bash
ZINC_HOST=your.server.ip
ZINC_PORT=22
ZINC_USER=root
```

The remote node needs: Zig 0.15.2+, Vulkan drivers, glslc, and a GGUF model file.

### Run the Loop

```bash
# Dry run — verifies SSH, rsync, build, and run (no AI agent)
bun loops/optimize_zinc.ts --dry-run

# Run 1 cycle
bun loops/optimize_zinc.ts --cycles 1

# Run overnight (infinite cycles, ctrl+c to stop)
bun loops/optimize_zinc.ts

# Custom model path
bun loops/optimize_zinc.ts --model-path /root/models/Qwen3-8B-Q4_K.gguf

# Resume a previous run
bun loops/optimize_zinc.ts --resume .zinc_optimize/2026-03-26T...
```

### How It Works

Each cycle:
1. **rsync** local source to the remote RDNA4 node
2. **Build** via `zig build` (compiles Zig + GLSL shaders)
3. **Run** `zinc --prompt ...` and capture output
4. **Analyze** — build errors? runtime crash? tok/s metrics?
5. **Spawn Claude** with full context (errors, history, RDNA4 constraints)
6. Claude edits local source files (one focused change per cycle)
7. **Verify** — rsync + rebuild + rerun
8. **Keep or revert** — git checkpoint, revert if regression

Two phases:
- **FIX** — resolve build errors, shader issues, Vulkan crashes
- **OPTIMIZE** — improve throughput once running (tok/s, bandwidth utilization)

Results are saved to `.zinc_optimize/` with full logs per cycle.

## RDNA4 Hardware Setup

For running ZINC on AMD RDNA4 GPUs:

```bash
# Required: enable cooperative matrix support
export RADV_PERFTEST=coop_matrix

# Recommended: disable GPU ECC for ~10% more bandwidth
# Add to /etc/default/grub:
GRUB_CMDLINE_LINUX_DEFAULT="... amdgpu.ras_enable=0"
# Then: update-grub && reboot
```

## Architecture

<p align="center">
  <img src="assets/architecture.svg" alt="ZINC Architecture" width="680">
</p>

## Benchmarks

Measured on **AMD Radeon AI PRO R9700** (RDNA4, 32 GB, 576 GB/s) with **Qwen3.5-35B-A3B Q4_K_XL** (20.7 GiB).
Same hardware, same model, same prompt (`"The capital of France is"`), 256 generated tokens. Benchmarked 2026-03-26.

### Decode throughput (tok/s, higher is better)

```
                 Qwen3.5-35B-A3B Q4_K — Decode (256 tokens)
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  llama.cpp   ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░  24.4 tok/s  │
  │  Vulkan                                                         │
  │                                                                 │
  │  ZINC        ██████████████████░░░░░░░░░░░░░░░░░░░  45.5 tok/s  │
  │  (current)                                         ▲ 1.87× ▲    │
  │                                                                 │
  │  ZINC        ███████████████████████████████████████ 110+ tok/s  │
  │  (target)                                                       │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

### Full comparison

| Metric | llama.cpp Vulkan | ZINC (current) | ZINC (target) |
|--------|:---:|:---:|:---:|
| **Decode** | 24.4 tok/s | **45.5 tok/s** (1.87x) | 110+ tok/s |
| **Prompt eval** | 55.6 tok/s | **2,655 tok/s** (pp10) | 2,800+ tok/s (pp512) |
| **Coherent output** | Yes (reasoning) | WIP | Yes |
| **RDNA4-tuned kernels** | No | Yes | Yes |
| **Native tokenizer** | Yes | Yes (BPE from GGUF) | Yes |
| **Continuous batching** | No | Phase 4 | Yes |
| **TurboQuant KV** | No | Phase 5 | Yes |

> **Note**: Both engines ran on the same GPU with the same model and prompt. ZINC prompt eval used 10 tokens (short prompt); llama.cpp reports its own prompt eval rate. ZINC output is not yet coherent (compute graph correctness is WIP) — throughput numbers reflect real GPU kernel execution time.

### Performance targets

#### Radeon AI PRO R9700 (32GB, 576 GB/s)
| Model | Quant | Single-req | 4-concurrent | Aggregate |
|-------|-------|------------|--------------|-----------|
| Qwen3-8B | Q4_K | 120+ tok/s | 110+ each | 440+ tok/s |
| Qwen3.5-35B-A3B | Q4_K | 110+ tok/s | 108+ each | 432+ tok/s |
| Llama-3.1-70B | Q4_K | 35+ tok/s | 32+ each | 128+ tok/s |

#### RX 9070 XT (16GB, 672 GB/s)
| Model | Quant | Single-req | 4-concurrent | Aggregate |
|-------|-------|------------|--------------|-----------|
| Qwen3-8B | Q4_K | 130+ tok/s | 120+ each | 480+ tok/s |

## Current Status

| Component | Status |
|-----------|--------|
| Vulkan infrastructure | Done |
| GGUF parser + model loader | Done |
| GPU detection (RDNA3/4) | Done |
| Native BPE tokenizer (from GGUF) | Done |
| GLSL compute shaders (16) | Done |
| Compute graph + architecture builders | Done |
| Forward pass (decode loop) | Running (2.7x llama.cpp, output correctness WIP) |
| HTTP server + OpenAI API | Phase 4 |
| Continuous batching | Phase 4 |
| TurboQuant KV compression | Phase 5 |

Validated on AMD Radeon AI PRO R9700 (RDNA4): Vulkan 1.3 init, GGUF parsing, 21GB model loaded to VRAM, 723-node MoE graph built, inference engine initialized.

## License

MIT
