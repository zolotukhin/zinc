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

ZINC takes the hardware these cards already have — 576 GB/s memory bandwidth, cooperative matrix units, 16–32 GB VRAM — and builds an inference engine that actually uses it.

**Hand-tuned for the hardware.** The GPU shaders are written specifically for RDNA4's memory hierarchy: wave64 dispatch, architecture-aware tiling, fused operations that cut redundant VRAM round-trips. Not a generic Vulkan backend that happens to run on AMD — built to hit 90%+ of theoretical memory bandwidth on the matmuls that dominate LLM decode.

**Built for serving, not demos.** Continuous batching with paged KV cache (same approach as vLLM) means multiple requests share the GPU without per-slot degradation. A single RX 9070 XT can serve 4+ concurrent users at full speed. TurboQuant KV compression shrinks cache memory 5x, doubling how many sessions fit before VRAM runs out.

**Drop-in compatible.** The API is OpenAI-compatible — point your existing client at it and it works. No ROCm, no CUDA, no driver stack to fight. One binary, one GPU, production inference on a $550 card.

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

All numbers measured on **AMD Radeon AI PRO R9700** (RDNA4, 32 GB, 576 GB/s) with **Qwen3.5-35B-A3B Q4_K_XL** (20.7 GiB, MoE 35B total / 3B active).
Same hardware, same model, same prompt (`"The capital of France is"`), 32 generated tokens. Benchmarked 2026-03-28.

### Decode throughput (tok/s, higher is better)

```
       Qwen3.5-35B-A3B Q4_K_XL — Decode, AI PRO R9700

  llama.cpp      ██████████████████████████████████████░░  107 tok/s
  (baseline)

  ZINC           █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  4.3 tok/s
  (current)

  ZINC           ██████████████████████████████████████░░  110+ tok/s
  (target)
```

### Comparison

| Metric | llama.cpp (baseline) | ZINC (current) | ZINC (target) |
|--------|:---:|:---:|:---:|
| **Decode** | 107 tok/s | 4.3 tok/s | 110+ tok/s |
| **Coherent output** | Yes | Yes | Yes |
| **BW utilization** | ~85% | 0.4% (2.1 GB/s) | 90%+ |
| **GPU syncs/token** | 1 | ~120 | 1–2 |
| **Flash attention** | Yes | Phase 3 | Yes |
| **RDNA4-tuned DMMV** | No | Yes | Yes |
| **Native BPE tokenizer** | Yes | Yes (from GGUF) | Yes |
| **Continuous batching** | Yes | Phase 4 | Yes |
| **TurboQuant KV** | No | Phase 5 | Yes |

> **Baseline setup**: llama-server (build `3306dba`) with `RADV_PERFTEST=coop_matrix`, `--flash-attn on`, `--mlock`, `-ngl 99`, `-ctk q8_0 -ctv q8_0`. Mesa 25.0.7, GECC disabled (`amdgpu.ras_enable=0`). See [RDNA4 Tuning Guide](docs/RDNA4_TUNING.md) and [AGENTS.md](AGENTS.md) for full setup and reproduction steps.

### Why 25x slower than llama.cpp

The gap is almost entirely **CPU-GPU synchronization overhead**, not GPU compute speed. Each decode token currently requires ~120 `vkQueueSubmit` + fence-wait round-trips:

- **MoE routing** requires GPU-to-CPU readback of router logits per layer (40 layers)
- **Shared expert gating** needs another readback per layer
- **End-of-layer submit** flushes the command buffer after each layer
- Each round-trip costs ~1–2 ms on RDNA4, totaling ~120–240 ms of pure sync overhead per token

At 256 ms/tok with 542 MB read per token, the GPU itself is idle >95% of the time. The fix is recording the full decode graph as a single command buffer with on-GPU MoE routing — the same approach llama.cpp uses.

### Output quality

ZINC produces **coherent, correct text** as of 2026-03-28:

```
Prompt:  "The capital of France is"
Output:  "Paris. The capital of Germany is Berlin. The capital of
          Italy is Rome. The capital of Spain is Madrid. The capital
          of Portugal is Lisbon..."
```

GPU-CPU numerical accuracy verified: embedding, RMS norm, DMMV (Q4_K/Q5_K/Q8_0), and LM head logits all match CPU reference within floating-point tolerance.

## Optimization Loop Results

The self-improving loop ran **186 cycles** across 6 sessions (March 27–28), with an AI agent iteratively fixing and optimizing the forward pass on real RDNA4 hardware.

### Performance progression

```
  tok/s    Qwen3.5-35B-A3B Q4_K_XL, AI PRO R9700
  4.5 ┤
       │                                        ╭─ 4.33 (best)
  4.0 ┤                  ╭──────────────────────╯── 3.9 avg
       │                  │
  3.5 ┤                  │
       │                  │
  3.0 ┤                  │
       │                  │
  2.5 ┤  ────────────────╯
       │  ~2.3 avg
  2.0 ┤
       │
  1.5 ┤
       │  ~1.4 avg
  1.0 ┤──╮
       └──┴──────────────────────────────────────── cycles
       0  43         87       102      145     186
       Mar 27 AM    Mar 27 PM Mar 28   Codex   Latest
```

| Phase | Cycles | tok/s | Key change |
|-------|-------:|------:|------------|
| First correct run | 43 | 1.2–2.4 | Forward pass executing all 40 layers + MoE + SSM |
| Bug fixes | 44 | 2.3 → 4.0 | Q4_K sub-block pairing fix (1.7x jump at cycle 20) |
| Coherent output | 15 | 3.8–4.1 | Q5_K element ordering fix → first correct text |
| Optimization plateau | 84 | 3.9–4.3 | Minor gains; bottleneck is sync overhead, not compute |

### Key bugs found by the loop

| Bug | Impact | Cycle |
|-----|--------|-------|
| Q4_K sub-block pairing: `(sp, sp+4)` → `(2*sp, 2*sp+1)` | 1.7x speedup (2.3 → 3.9 tok/s) | 16-50/C19 |
| Q5_K element ordering: interleaved → contiguous sub-blocks | Garbage → coherent output | 06-38/C05 |
| Q8_0/F16 wave32 subgroup: lost half the dot product | Partial correctness | 16-50/C03 |
| Shared expert intermediate dim: 1408 → 5632 | Wrong FFN output | 16-50/C26 |
| Shared expert sigmoid gating: was TODO, now implemented | Missing computation | 16-50/C08 |
| SSM conv1d ordering: convolve before state update | Double-counting input | 16-50/C06 |
| Q4_K SPEC_K: fixed constant → push-constant K | Wrong for non-hidden-dim projections | 16-50/C09 |
| SSM delta-net K/Q head mapping: division → modular | Wrong head assignment | 02-57/C06 |
| attn_out_buf overflow: `q_dim*4` → `q_dim*2*4` | Buffer overwrite | 16-50/C14 |

46 changes kept out of 186 cycles (25% acceptance rate).

## Current Status

| Component | Status |
|-----------|--------|
| Vulkan infrastructure | Done |
| GGUF parser + model loader | Done |
| GPU detection (RDNA3/4) | Done |
| Native BPE tokenizer (from GGUF) | Done |
| GLSL compute shaders (16) | Done |
| Compute graph + architecture builders | Done |
| Forward pass (decode loop) | Working — 4.3 tok/s, coherent output |
| Single command buffer decode | Next — eliminate 120 syncs/token overhead |
| HTTP server + OpenAI API | Phase 4 |
| Continuous batching | Phase 4 |
| TurboQuant KV compression | Phase 5 |

Validated on AMD Radeon AI PRO R9700 (RDNA4): Vulkan 1.3 init, GGUF parsing, 21 GB model loaded to VRAM, 723-node MoE graph built, coherent inference output verified against CPU reference.

## Next Steps

The path from 4.3 to 110+ tok/s:

1. **Single command buffer** — Record the full 40-layer decode as one Vulkan submission. Move MoE routing to GPU (softmax + top-k in shader). Eliminates ~120 round-trips per token. Expected: 20–40x improvement.
2. **Pre-recorded command buffer** — Record once, replay per token with updated push constants. Eliminates per-token descriptor allocation and command recording overhead.
3. **Flash attention** — Replace per-head DMMV attention with the existing `flash_attn.comp` shader. Reduces attention memory traffic ~4x.
4. **Profiling + kernel tuning** — With sync overhead gone, profile actual GPU kernel time. Tune DMMV tile sizes, shared memory usage, and occupancy for RDNA4.

## License

MIT
