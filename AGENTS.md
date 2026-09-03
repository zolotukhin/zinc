# AGENTS.md — ZINC Inference Engine

Instructions for AI coding agents working on this repository.

## Commands

```bash
# Build (shaders compile on Linux only; macOS skips GPU inference)
zig build -Doptimize=ReleaseFast

# Run inference
ZINC_DEBUG=1 ./zig-out/bin/zinc -m model.gguf --prompt "Hello" [-d device_id] [--kv-quant 3] [--debug]
./zig-out/bin/zinc --model-id qwen35-9b-q4k-m --prompt "Hello" [--chat]

# Run unit tests
zig build test

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

- **Zig 0.15.2+** — host code, build system
- **GLSL 460** — compute shaders compiled to SPIR-V via `glslc` (Vulkan backend)
- **MSL** — Metal Shading Language compute shaders (Apple Silicon backend)
- **Vulkan 1.3** — GPU API for AMD RDNA3/RDNA4
- **Metal** — GPU API for Apple Silicon (M1–M5)
- **GGUF** — model format (parsed natively in Zig)

### Supported Architectures
- **Qwen 2/3 family** — dense, MoE, and hybrid attention/SSM variants
- **Mistral/Llama, Mamba/Jamba, Gemma, GPT-OSS, and Muse Glimmer**

## Project Structure

```
src/
├── main.zig                     # CLI entry, arg parsing, server startup, chat subcommand
├── bench_hot_decode.zig         # Hot decode microbenchmark binary
├── bench_support.zig            # Shared helpers for Metal benchmarks
├── regression_tests.zig         # Source-level regression guards
├── zig-struct-analyzer.zig      # Zig API/doc structure extraction helper
├── compute/
│   ├── forward.zig              # Vulkan inference engine — prefill + decode loop
│   ├── forward_metal.zig        # Metal inference engine — prefill + decode loop
│   ├── forward_cuda*.zig        # CUDA inference engines
│   ├── forward_zinc_rt.zig      # ZINC_RT inference engine
│   ├── dmmv.zig                 # DMMV dispatch (quantized matmul-vec)
│   ├── elementwise.zig          # Fused elementwise ops (RMS norm, SwiGLU, etc.)
│   ├── attention.zig            # Flash attention dispatch
│   ├── argmax.zig               # Argmax / sampling dispatch
│   └── graph.zig                # Decode graph builder and exporter
├── model/
│   ├── tokenizer.zig            # BPE tokenizer, chat templates, thinking toggle
│   ├── catalog.zig              # Managed model catalog with thinking_stable flag
│   ├── gguf.zig                 # GGUF file parser and tensor metadata
│   ├── loader.zig               # Model loader (Vulkan — mmap + DMA to VRAM)
│   ├── loader_metal.zig         # Model loader (Metal — zero-copy mmap)
│   ├── loader_cuda.zig          # CUDA model loader
│   ├── architecture.zig         # Architecture detection (Qwen, MoE, SSM, etc.)
│   ├── config.zig               # Model configuration from GGUF metadata
│   └── managed.zig              # Managed model download, install, activation
├── server/
│   ├── routes.zig               # OpenAI-compatible API, streaming, stop detection
│   ├── chat.html                # Built-in chat UI (embedded at compile time)
│   ├── http.zig                 # HTTP server and connection handling
│   ├── model_manager.zig        # Hot model switching and catalog view
│   ├── runtime.zig              # Backend runtime dispatch
│   └── session.zig              # Chat session state
├── vulkan/
│   ├── instance.zig             # Vulkan instance and device init
│   ├── pipeline.zig             # Compute pipeline and shader loading
│   ├── buffer.zig               # GPU buffer allocation and transfers
│   ├── command.zig              # Command buffer recording and submission
│   ├── gpu_detect.zig           # GPU vendor/capability detection
│   └── vk.zig                   # Vulkan C API bindings
├── metal/
│   ├── c.zig                    # Shared C shim import for Metal bindings
│   ├── device.zig               # Metal device init and capability query
│   ├── pipeline.zig             # MSL compute pipeline compilation
│   ├── buffer.zig               # Metal buffer management
│   ├── command.zig              # Command buffer and encoder
│   ├── shim.h                   # C header exposed to Zig
│   └── shim.m                   # Objective-C shim (Metal.framework bridge)
├── cuda/                        # CUDA device, buffers, pipeline, and kernels
├── zinc_rt/                     # Native GPU runtime, IR, rings, and ISA
├── gpu/
│   ├── interface.zig            # Backend abstraction (Vulkan vs Metal)
│   ├── memory_plan.zig          # Cross-backend memory planning
│   └── process_lock.zig         # Cross-process GPU reservation lock
├── scheduler/
│   ├── request.zig              # Request/response scheduler types
│   ├── scheduler.zig            # Request scheduling
│   └── kv_cache.zig             # KV cache management
├── diagnostics.zig              # --check system diagnostics (Vulkan)
├── diagnostics_metal.zig        # --check system diagnostics (Metal)
├── shaders/
│   ├── *.comp                   # GLSL compute shaders (Vulkan/SPIR-V)
│   └── metal/*.metal            # MSL compute shaders (Apple Silicon)

benchmarks/
├── bandwidth.zig                # DMMV bandwidth utilization benchmark
├── dispatch.zig                 # Vulkan dispatch overhead benchmark
├── dispatch_overhead.c          # Vulkan dispatch overhead C helper
├── metal_inference.zig          # Metal inference benchmark
└── metal_q8_shapes.zig          # Exact-shape Metal q8 benchmark suite

loops/                           # Self-improving optimization loops
├── guided_change.ts             # Guided single-change optimization loop
├── guided_change.test.ts        # Tests for guided_change
├── implement_metal.ts           # Metal implementation loop
├── implement_metal.test.ts      # Tests for implement_metal
├── optimize_llm_tps.ts          # LLM-throughput optimization loop
├── optimize_llm_tps.test.ts     # Tests for optimize_llm_tps
├── optimize_perf.ts             # Performance optimization loop
├── optimize_perf.test.ts        # Tests for optimize_perf
├── optimize_zinc.ts             # ZINC loop: rsync → build → run → agent → keep/revert
├── optimize_zinc.test.ts        # Tests for optimize_zinc
└── zinc_rt_autopilot.ts         # Overnight A/B vs legacy_vulkan; drives docs/ZINC_RT_DESIGN.md

docs/                            # Technical documentation (published to site)
├── API.md                       # OpenAI-compatible API spec
├── APPLE_METAL_REFERENCE.md     # Metal/MSL kernel reference
├── APPLE_SILICON_METAL_ENABLEMENT.md # Metal port implementation notes
├── APPLE_SILICON_REFERENCE.md   # Apple Silicon M1–M5 reference
├── DEVELOPMENT.md               # Development guide (canonical dev reference)
├── GETTING_STARTED.md           # First run guide
├── AMD_GPU_REFERENCE.md         # RDNA3/RDNA4 hardware reference
├── HARDWARE_REQUIREMENTS.md     # GPU and host sizing guidance
├── METAL_PERFORMANCE_PLAN.md    # Metal performance work plan
├── RDNA4_TUNING.md              # RDNA4-specific optimizations
├── ROADMAP.md                   # Project roadmap
├── RUNNING_ZINC.md              # CLI usage and server mode
├── SPEC.md                      # Architecture overview
├── TURBOQUANT_SPEC.md           # TurboQuant KV cache compression spec
└── ZINC_RT_DESIGN.md            # ZINC's own GPU runtime — replaces Vulkan; design + milestone plan

site/                            # Astro website + docs frontend (zolotukhin.ai)
├── src/components/              # Shared Astro UI components
├── src/content/posts/           # Blog post markdown sources
├── src/layouts/                 # Page layouts
├── src/lib/                     # Search/docs/data loaders and tests
├── src/pages/                   # Site routes, docs, and API-ish endpoints
├── src/styles/                  # Global site styling
└── public/                      # Static assets served by Astro

specs/                           # Historical feature design artifacts
├── 001-zinc-inference-engine/
├── 002-microblog-zolotukhin-ai/
├── 003-decode-performance/
├── 004-openai-api-server/
└── 005-apple-silicon-inference/

tools/                           # API benchmark, HTML tooling, graph/report helpers
├── benchmark_api.mjs
├── chat.html
├── dump_struct_layouts.zig
├── print_test_summary.ts
└── render_graph_report.ts

tests/                           # TypeScript integration/smoke tests
├── chat_ui_markdown.test.ts
├── test_openai_sdk.ts
├── test_openai_sdk.test.ts
├── test_qwen_smoke.ts
└── test_qwen_smoke.test.ts

assets/                          # Shared repo/media assets
research/                        # Analysis notes and external comparisons
scripts/                         # Deployment scripts
writing/                         # Draft writing and publishing notes
```

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
- Run `zig build test` before considering work complete
- Validate GPU kernels against llama.cpp reference outputs
- Use push constants (not UBOs) for per-dispatch parameters in shaders
- Keep shader local_size_x = 64 (RDNA4 wave64)

### Ask first
- Changing the compute graph IR (`graph.zig` OpType enum)
- Adding new model architectures to `architecture.zig`
- Modifying Vulkan initialization or device selection
- Changes to GGUF parsing that could break existing model loading

### Never
- Commit `.env`, credentials, private IPs, private hostnames, SSH aliases, private ports, GPU UUIDs, device serials, or host-specific usernames in paths
- Hard-code remote endpoints or machine selectors in scripts, docs, prompts, tests, or generated artifacts; use `.env`, environment variables (`ZINC_HOST`, `ZINC_USER`, `ZINC_PORT`, `ZINC_GPU`, `ZINC_GPU_4090`, etc.), and placeholders such as `<host>` or `<cuda-device>` instead
- Treat loopback (`127.0.0.1`, `localhost`, `0.0.0.0`) or public project/service URLs as leaks, but do re-check any non-loopback address or private-looking hostname before committing
- Modify `.spv` binaries directly — always recompile from `.comp` source
- Add runtime dependencies beyond Vulkan and system libc
- Use wave32 without benchmarking against wave64 first

## Remote Test Node

An RDNA4 test node (AMD Radeon AI PRO R9700, 32GB, 576 GB/s) is available via SSH. Credentials are in `.env` (gitignored) as `ZINC_HOST`, `ZINC_USER`, `ZINC_PORT`.

## Running Benchmarks

### Baseline: 107 tok/s (2026-03-26)

The reference baseline is llama.cpp server on the RDNA4 test node with this exact configuration. All ZINC numbers are compared against this.

Headline RDNA comparisons must use the fair server-vs-server harness in
`tools/performance_suite.mjs`: one reusable ZINC server per model, one reusable
llama.cpp server per model, same GGUF, same scenario matrix, same warmup/run
count, and server-side prefill/decode timings. Do not compare a one-shot ZINC
CLI run against a warmed llama.cpp server when deciding whether a metric is
beaten. CLI runs are diagnostics only.

**Model**: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (20.7 GiB, MoE 35B/3B active)
**Baseline result**: 107 tok/s decode (with reasoning), 223 tok/s prefill

### Test node setup (critical for reproducing baseline)

```bash
# 1. Mesa must be 25.0.7 (25.2.8 causes ~14% RADV regression)
dpkg -l mesa-vulkan-drivers  # should show 25.0.7-0ubuntu0.24.04.2
# Pinned in /etc/apt/preferences.d/mesa-pin to prevent auto-upgrade

# 2. GECC disabled (amdgpu.ras_enable=0 in /etc/default/grub)
cat /sys/module/amdgpu/parameters/ras_enable  # should show 0

# 3. RADV_PERFTEST=coop_matrix set in llama-server.service
#    Without this, cooperative matrix is disabled → scalar fallback

# 4. llama.cpp build 3306dba, built with:
#    cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
#      -DCMAKE_CXX_FLAGS='-O3 -march=znver4' -DCMAKE_C_FLAGS='-O3 -march=znver4'

# 5. Server flags (in /etc/systemd/system/llama-server.service):
#    -ngl 99 --device Vulkan<N> --parallel 4 -c 32768
#    -ctk q8_0 -ctv q8_0 -b 4096 -ub 1024 --mlock --flash-attn on
#    Pick the discrete GPU index from `vulkaninfo --summary`; on mixed APU+dGPU
#    nodes this may be Vulkan1 rather than Vulkan0.
```

### Measure llama.cpp baseline

```bash
source .env

# Start server (if not running)
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "systemctl start llama-server && sleep 15"

# Warmup + 3 benchmark runs via OpenAI API
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST '
  curl -s http://localhost:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"q\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" > /dev/null
  for i in 1 2 3; do
    out=$(curl -s http://localhost:8088/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"q\",\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}],\"max_tokens\":256,\"stream\":false}" \
    )
    gen=$(printf "%s" "$out" | jq -r ".timings.predicted_per_second // 0")
    prompt=$(printf "%s" "$out" | jq -r ".timings.prompt_per_second // 0")
    printf "Run %d: gen %s tok/s | prompt %s tok/s\n" "$i" "$gen" "$prompt"
  done
'
# Expected: ~107 tok/s generation, ~220 tok/s prompt (runs 2-3, after warmup)
```

### Measure ZINC CLI diagnostics

```bash
source .env

# Sync source to test node
rsync -az --delete --exclude '.zig-cache' --exclude 'zig-out' --exclude 'node_modules' \
  --exclude '.DS_Store' --exclude 'site' \
  -e "ssh -p $ZINC_PORT" . $ZINC_USER@$ZINC_HOST:/root/zinc/

# Build and run
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "cd /root/zinc && zig build -Doptimize=ReleaseFast && \
  RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
  -m /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt 'The capital of France is'"

# Key output lines:
#   info(forward): Prefill complete: N tokens in X ms (Y tok/s)
#   info(forward): Generated N tokens in X ms — Y tok/s (Z ms/tok)
```

Use this CLI path for quick engine checks only. For a fair RDNA ZINC-vs-llama.cpp
claim, run the performance suite so both engines are measured through reusable
servers:

```bash
source .env

bun tools/performance_suite.mjs \
  --target rdna \
  --phase all \
  --rdna-sync \
  --rdna-build \
  --rdna-start-llama \
  --runs 3 \
  --warmup 1 \
  --no-site-write \
  --output /tmp/zinc-rdna-suite-$(date +%Y%m%d-%H%M%S).json
```

### Measure ZINC API endpoints

Use the HTTP benchmarks when you need end-to-end API latency, queueing behavior, or to compare the chat endpoint against the raw completions path.

Important caveats before you trust the numbers:

1. Bench a clean node. Other `zinc`, `llama-server`, and `llama-cli` processes on the RDNA4 host will contaminate both latency and throughput.
2. `POST /v1/chat/completions` is an end-user latency benchmark, not a pure decode-throughput benchmark. The chat route applies templates and stop handling, so many prompts stop after only a handful of tokens.
3. Use `POST /v1/completions` for sustained HTTP decode throughput. It avoids chat stop-sequence behavior and is the closest HTTP-side equivalent to the CLI `--prompt` path.
4. ZINC server generation is still serialized. With `concurrency > 1`, aggregate throughput stays roughly flat while per-request latency grows because requests queue behind one active decode.

Clean-server setup:

```bash
source .env

# 1. Stop stale GPU users on the test node.
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  pkill -f 'zig-out/bin/zinc' || true; \
  pkill -f 'llama-server' || true; \
  pkill -f 'llama-cli' || true"

# 2. Sync, build, and restart one clean ZINC server on :9090.
rsync -az --delete --exclude '.zig-cache' --exclude 'zig-out' --exclude 'node_modules' \
  --exclude '.DS_Store' --exclude 'site' \
  -e "ssh -p $ZINC_PORT" . $ZINC_USER@$ZINC_HOST:/root/zinc/

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && zig build -Doptimize=ReleaseFast && \
  nohup env RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
    -m /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --port 9090 >/tmp/zinc_9090.log 2>&1 < /dev/null &"

# 3. Wait for health.
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  until curl -fsS http://127.0.0.1:9090/health >/dev/null; do sleep 1; done; \
  curl -sS http://127.0.0.1:9090/health"
```

Chat-endpoint latency matrix:

```bash
source .env

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && \
  /root/.bun/bin/bun tools/benchmark_api.mjs \
    --base http://127.0.0.1:9090/v1 \
    --mode chat \
    --output /tmp/zinc_api_chat_benchmark.json"
```

Raw sustained-throughput benchmark:

```bash
source .env

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && \
  /root/.bun/bin/bun tools/benchmark_api.mjs \
    --base http://127.0.0.1:9090/v1 \
    --mode raw \
    --output /tmp/zinc_api_raw_benchmark.json"
```

### Measure hot decode kernels directly

Use the dedicated microbenchmark when whole-model decode says “MoE”, “shared
expert”, or `ssm_delta_net` is hot and you need exact per-kernel numbers plus
`RADV_DEBUG=shaderstats` feedback.

Important caveat:

- the current hot-bench path rotates across multiple buffer sets to reduce the
  worst cache-hot bias
- still treat its reported GB/s as a kernel-comparison signal, not as the final
  whole-model DRAM bandwidth number

```bash
source .env

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && \
  zig build hot-bench -Doptimize=ReleaseFast -- \
    --model /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --iterations 200 --warmup 25"
```

Focused single-case runs:

```bash
source .env

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && \
  zig build hot-bench -Doptimize=ReleaseFast -- \
    --model /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --case q8_router"

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && \
  zig build hot-bench -Doptimize=ReleaseFast -- \
    --model /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --case q8_shared_gate_up"

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && \
  RADV_DEBUG=shaderstats zig build hot-bench -Doptimize=ReleaseFast -- \
    --model /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --case ssm_delta"
```

Available cases right now:

- `q8_router`
- `q8_shared_gate_up`
- `q8_shared_down`
- `q8_ssm_out`
- `ssm_delta`

### Troubleshooting performance

If llama.cpp baseline drops below ~100 tok/s, check in order:
1. **Mesa version** — `dpkg -l mesa-vulkan-drivers` must show 25.0.7 (not 25.2.8)
2. **GECC** — `cat /sys/module/amdgpu/parameters/ras_enable` must show 0
3. **coop_matrix** — server log must show `matrix cores: KHR_coopmat`
4. **Reboot** — Mesa/driver changes need a reboot to take full effect
5. **Dirty benchmark node** — stop stray `zinc` / `llama-*` processes before comparing runs
6. **Wrong endpoint for the question** — use `/v1/chat/completions` for chat latency and queueing, `/v1/completions` for sustained HTTP decode throughput
7. **Early chat stops** — if chat completions are ending after a handful of tokens, change the prompt or switch to `/v1/completions`; otherwise the reported completion TPS is mostly prompt+HTTP overhead

## Code Navigation

The source changes frequently, so this section intentionally avoids line numbers,
buffer inventories, kernel counts, and tuning thresholds. Search for the named
types or functions before editing.

### Entrypoints and API

| Area | Start here |
|------|------------|
| CLI, argument parsing, server startup | `src/main.zig` |
| HTTP transport and OpenAI-compatible routes | `src/server/http.zig`, `src/server/routes.zig` |
| Backend selection and model switching | `src/server/runtime.zig`, `src/server/model_manager*.zig` |
| Built-in chat UI | `src/server/chat.html` |
| Tokenization and chat templates | `src/model/tokenizer.zig` |

### Models and inference

| Area | Start here |
|------|------------|
| GGUF metadata and tensors | `src/model/gguf.zig`, `src/model/config.zig` |
| Architecture detection | `src/model/architecture.zig` |
| Managed model catalog/cache | `src/model/catalog.zig`, `src/model/managed.zig` |
| Vulkan model loading | `src/model/loader.zig` |
| Metal and CUDA model loading | `src/model/loader_metal.zig`, `src/model/loader_cuda.zig` |
| Vulkan inference | `src/compute/forward.zig` |
| Metal inference | `src/compute/forward_metal.zig` |
| CUDA inference | `src/compute/forward_cuda.zig`, `src/compute/forward_cuda_gemma.zig` |
| ZINC_RT inference | `src/compute/forward_zinc_rt.zig`, `src/zinc_rt/` |

The conceptual request path is: tokenize → load/activate model → prefill prompt →
decode tokens → sample → detokenize/stream. Architecture-specific attention, SSM,
dense FFN, and MoE branches live in the active backend's forward implementation.

### GPU operations

| Area | Start here |
|------|------------|
| Compute graph | `src/compute/graph.zig` |
| Quantized matrix-vector dispatch | `src/compute/dmmv.zig` |
| Attention, elementwise ops, sampling | `src/compute/attention.zig`, `src/compute/elementwise.zig`, `src/compute/argmax.zig` |
| Vulkan infrastructure | `src/vulkan/`, `src/shaders/*.comp` |
| Metal infrastructure | `src/metal/`, `src/shaders/metal/*.metal` |
| CUDA infrastructure | `src/cuda/` |
| Backend abstraction and memory planning | `src/gpu/` |

### Tests and performance tooling

| Area | Start here |
|------|------------|
| Zig regression tests | `src/regression_tests.zig` |
| Integration and API tests | `tests/` |
| Microbenchmarks | `benchmarks/`, `src/bench_hot_decode.zig` |
| Fair cross-backend suite | `tools/performance_suite.mjs` |
| Optimization harnesses | `loops/` |

Treat implementation source and command `--help` output as authoritative for
current environment flags, thresholds, supported formats, and benchmark cases.
Update this map only when ownership or major entrypoints move.
