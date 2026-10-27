# AGENTS.md — ZINC Inference Engine

Instructions for AI coding agents working on this repository.

## Commands

```bash
# Build Vulkan on Linux or Metal on macOS
./scripts/zig.sh build -Doptimize=ReleaseFast

# Run inference
ZINC_DEBUG=1 ./zig-out/bin/zinc -m model.gguf --prompt "Hello" [-d device_id] [--kv-quant 3] [--debug]
./zig-out/bin/zinc --model-id qwen35-9b-q4k-m --prompt "Hello" [--chat]

# Run unit tests
./scripts/zig.sh build test

# Compile shaders manually (requires glslc / shaderc)
glslc --target-env=vulkan1.3 -O -o out.spv src/shaders/name.comp
```

### Agent harness scripts (`loops/`)

Run-to-completion and autonomous harnesses that spawn `claude` or `codex` subagents. See file headers for full flags.

```bash
# One-shot: run a single prompt to completion with pre/post benchmarks,
# auto-retry on regressions (up to 3x) before reverting.
bun loops/guided_change.ts --prompt "..."          # inline prompt
bun loops/guided_change.ts --prompt-file plan.md   # from file
echo "..." | bun loops/guided_change.ts            # from stdin

# Autonomous multi-cycle loops (run until killed):
bun loops/optimize_zinc.ts          # rsync → build → run → agent → keep/revert on RDNA4 test node
bun loops/optimize_llm_tps.ts --agent claude "..."  # iterate on llama.cpp throughput
bun loops/optimize_perf.ts --effort N               # execute loops/efforts/MULTI_HOUR_EFFORT_N.md
bun loops/implement_metal.ts        # iteratively build out the Metal backend
bun loops/zinc_rt_autopilot.ts      # overnight A/B: legacy_vulkan vs ZINC_RT on RDNA4
                                    # see docs/ZINC_RT_DESIGN.md for the project this drives
```

### Managed models and cache

For local Apple Silicon work, always prefer ZINC's managed model cache over ad hoc GGUF paths.

```bash
# Inspect the managed catalog for the current GPU profile
./zig-out/bin/zinc model list
./zig-out/bin/zinc model list --all

# Pull a managed model into the default local cache
./zig-out/bin/zinc model pull qwen35-9b-q4k-m

# Run via the managed model id instead of a handwritten GGUF path
./zig-out/bin/zinc --model-id qwen35-9b-q4k-m --prompt "What is the capital of France?" --chat
```

Default local managed cache layout:

```text
~/Library/Caches/zinc/models/models/<model-id>/model.gguf
```

Agent policy:

- On local Metal, use `./zig-out/bin/zinc model pull <id>` to fetch catalog models.
- On local Metal, prefer `--model-id <id>` or the default managed cache path over arbitrary `/Users/.../*.gguf` paths.
- Do not copy large GGUFs into one-off local locations when a catalog id exists.
- If a needed local model is missing, install it into the managed cache or explicitly explain why that is not possible.
- RDNA node runs are the exception: the benchmark node still uses the canonical `/root/models/...` GGUF paths unless the repo explicitly migrates that workflow too.

## Tech Stack

- **Zig 0.17.0** — host code, build system
- **Bun 1.4.4 minimum** — JavaScript tooling; `.bun-version` pins CI. Use `bun fmt` for supported files, with configuration in `bunfig.toml`. Do not add Prettier.
- **GLSL 460** — compute shaders compiled to SPIR-V via `glslc` (Vulkan backend)
- **MSL** — Metal Shading Language compute shaders (Apple Silicon backend)
- **Vulkan 1.3** — GPU API for AMD RDNA3/RDNA4
- **Metal** — GPU API for Apple Silicon (M1–M5)
- **GGUF** — model format (parsed natively in Zig)

### Supported Architectures

- **Qwen 2/3 family** — dense, MoE, and hybrid attention/SSM variants
- **Mistral/Llama, Mamba/Jamba, Gemma, GPT-OSS, and Muse Glimmer**

## Code navigation

Read [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md) for source ownership, backend
entry points, and test locations. Implementation source and command `--help`
output are authoritative for current flags, thresholds, and supported formats.
Update the map when ownership or major entry points move.

## Key Architecture Decisions

- **Static graph pre-recording**: decode graph built once per model arch, command buffer recorded once, replayed per token with updated push constants/descriptors
- **Quantization isolated in shaders**: graph nodes use `dmmv` ops; dispatcher selects Q4_K/Q8_0/F16 pipeline at runtime
- **GPU auto-tuning**: `gpu_detect.zig` classifies hardware and derives wave size, tile sizes, cache parameters — no manual config needed
- **Paged KV cache**: 16-token pages (vLLM-style) via page table in flash attention shader
- **Fused kernels**: RMS_NORM_MUL, SWIGLU, ROPE_FUSED eliminate intermediate memory traffic

## Code Style

- **Zig**: follow standard Zig conventions, `zig fmt` for formatting
- **GLSL**: `#version 460`, `layout(local_size_x = 64)` default (wave64 for RDNA), push constants for per-dispatch params, storage buffers for data
- Keep shader workgroup size at 64 unless there's a measured reason to change it

## Boundaries

### Always

- Run `./scripts/zig.sh build test` before considering work complete
- Validate GPU kernels against llama.cpp reference outputs
- Use push constants (not UBOs) for per-dispatch parameters in shaders
- Keep shader local_size_x = 64 (RDNA4 wave64)

### Ask first

- Any run on the workstation's own GPU (Metal target, Metal debugging, local model loads); see "GPU Workload Placement"
- Changing the compute graph IR (`graph.zig` OpType enum)
- Adding new model architectures to `architecture.zig`
- Modifying Vulkan initialization or device selection
- Changes to GGUF parsing that could break existing model loading

### Never

- Load a model, start an inference server, or run a benchmark on the developer workstation without the user's explicit go-ahead for that run; all GPU workloads go to the remote test node (see "GPU Workload Placement")
- Leave a spawned `zinc` or `llama-server` process running after the command that started it returns
- Commit `.env`, credentials, private IPs, private hostnames, SSH aliases, private ports, GPU UUIDs, device serials, or host-specific usernames in paths
- Hard-code remote endpoints or machine selectors in scripts, docs, prompts, tests, or generated artifacts; use `.env`, environment variables (`ZINC_HOST`, `ZINC_USER`, `ZINC_PORT`, `ZINC_GPU`, `ZINC_GPU_4090`, etc.), and placeholders such as `<host>` or `<cuda-device>` instead
- Treat loopback (`127.0.0.1`, `localhost`, `0.0.0.0`) or public project/service URLs as leaks, but do re-check any non-loopback address or private-looking hostname before committing
- Modify `.spv` binaries directly — always recompile from `.comp` source
- Add runtime dependencies beyond Vulkan and system libc
- Use wave32 without benchmarking against wave64 first

## GPU Workload Placement (hard rule)

The developer workstation (the machine the agent's shell runs on, including
Apple Silicon Macs) is a shared desktop, not a benchmark box. In September 2026
an agent debugging a Metal MoE prefill loaded a 17 GB GGUF in ZINC while a
`llama-server` it had started earlier still held the same model. WindowServer
starved of GPU time, macOS panicked with `userspace watchdog timeout`, and the
workstation rebooted mid-session.

**All model and GPU workloads run on the remote test node**, over
`ssh -p "$ZINC_PORT" "$ZINC_USER@$ZINC_HOST"` (values from `.env`). That covers:

- `zinc` CLI or server runs, including `--model-id` runs, validate/bisect/profile
  env modes, and anything else that loads a GGUF
- every llama.cpp binary (`llama-server`, `llama-cli`, `llama-bench`,
  `llama-eval-callback`, ...)
- `bun tools/performance_suite.mjs` with `--target metal` or `--target both`
  (`--target rdna` is fine locally: `bun` only drives the node over SSH)
- `bun test` suites or smoke tests that start a server or load a model
- `zinc model pull` for benchmarking purposes, and `--metal-pull-missing`

**Allowed on the workstation:** `./scripts/zig.sh build`, `zig fmt`, `./scripts/zig.sh build test` cases that
do not open a model file, shader compilation, git, the site build, and
`wrangler` deploys.

**Metal-only work** (a `forward_metal.zig` bug, the Metal perf-suite target, a
Metal kernel) needs the user's explicit go-ahead for that specific run, in the
current conversation. A general request to "fix", "benchmark", or "continue" is
not that go-ahead. Without it, stop and ask; record findings so far instead of
running anything.

**Preflight before any command that could touch a GPU** (check all three, every
time):

1. The inference part of the command is inside the `ssh ... "$ZINC_HOST"`
   invocation. If `./zig-out/bin/zinc`, `llama-*`, or a model path under
   `~/Library/Caches/zinc` would execute locally, do not run it.
2. No GPU process from this session is still alive where it will run:
   `for n in zinc llama-server llama-cli llama-bench; do pgrep -x "$n"; done`
   must list nothing of yours (the node's long-running service is the one
   exception you manage). Match exact process names with `pgrep -x`; a
   `pgrep -f` pattern also matches the `ssh`/`bash -c` command line that carries
   the check and the inference command, so it always fires.
3. The command stops what it starts. Launch servers inside the same command that
   uses them and stop them by PID before it returns (`trap 'kill $PID' EXIT`);
   never leave a spawned server running between tool calls on any machine.

Even with approval to use a local GPU, run one GPU process at a time and never
two copies of a multi-GB model.

## Remote Test Node

An RDNA4 test node (AMD Radeon AI PRO R9700, 32GB, 576 GB/s) is available via SSH. Credentials are in `.env` (gitignored) as `ZINC_HOST`, `ZINC_USER`, `ZINC_PORT`.

## Running benchmarks

Before benchmark or performance work, read
[docs/AGENT_BENCHMARKS.md](docs/AGENT_BENCHMARKS.md) and the active backend guide.
Use `tools/performance_suite.mjs` for headline RDNA comparisons: one reusable
server per engine/model, identical GGUF and scenarios, matching warmups and run
counts. One-shot CLI runs are diagnostics only.
