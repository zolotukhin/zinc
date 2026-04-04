# AGENTS.md — ZINC Inference Engine

Instructions for AI coding agents working on this repository.

## Commands

```bash
# Build (shaders compile on Linux only; macOS skips GPU inference)
zig build -Doptimize=ReleaseFast

# Run inference
ZINC_DEBUG=1 ./zig-out/bin/zinc -m model.gguf --prompt "Hello" [-d device_id] [--kv-quant 3] [--debug]

# Run unit tests
zig build test

# Compile shaders manually (requires glslc / shaderc)
glslc --target-env=vulkan1.3 -O -o out.spv src/shaders/name.comp
```

## Tech Stack

- **Zig 0.15.2+** — host code, build system
- **GLSL 460** — compute shaders compiled to SPIR-V via `glslc`
- **Vulkan 1.3** — GPU API (direct C ABI calls via `@cImport`)
- **GGUF** — model format (parsed natively in Zig)

## Project Structure

```
src/
├── main.zig                     # CLI entry, arg parsing, server startup, chat subcommand
├── compute/
│   ├── forward.zig              # Vulkan inference engine — prefill + decode loop
│   ├── forward_metal.zig        # Metal inference engine — prefill + decode loop
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
│   ├── architecture.zig         # Architecture detection (Qwen, MoE, SSM, etc.)
│   ├── config.zig               # Model configuration from GGUF metadata
│   └── managed.zig              # Managed model download, install, activation
├── server/
│   ├── routes.zig               # OpenAI-compatible API, streaming, stop detection
│   ├── chat.html                # Built-in chat UI (embedded at compile time)
│   ├── http.zig                 # HTTP server and connection handling
│   ├── model_manager.zig        # Hot model switching and catalog view
│   ├── runtime.zig              # Backend runtime dispatch (Vulkan vs Metal)
│   └── session.zig              # Chat session state
├── vulkan/
│   ├── instance.zig             # Vulkan instance and device init
│   ├── pipeline.zig             # Compute pipeline and shader loading
│   ├── buffer.zig               # GPU buffer allocation and transfers
│   ├── command.zig              # Command buffer recording and submission
│   ├── gpu_detect.zig           # GPU vendor/capability detection
│   └── vk.zig                   # Vulkan C API bindings
├── metal/
│   ├── device.zig               # Metal device init and capability query
│   ├── pipeline.zig             # MSL compute pipeline compilation
│   ├── buffer.zig               # Metal buffer management
│   ├── command.zig              # Command buffer and encoder
│   └── shim.m                   # Objective-C shim (Metal.framework bridge)
├── gpu/
│   └── interface.zig            # Backend abstraction (Vulkan vs Metal)
├── scheduler/
│   ├── scheduler.zig            # Request scheduling
│   └── kv_cache.zig             # KV cache management
├── diagnostics.zig              # --check system diagnostics (Vulkan)
├── diagnostics_metal.zig        # --check system diagnostics (Metal)
├── shaders/
│   ├── *.comp                   # GLSL compute shaders (Vulkan/SPIR-V) — 24 shaders
│   └── metal/*.metal            # MSL compute shaders (Apple Silicon) — 31 shaders

benchmarks/
├── bandwidth.zig                # DMMV bandwidth utilization benchmark
├── dispatch.zig                 # Vulkan dispatch overhead benchmark
└── metal_inference.zig          # Metal inference benchmark

loops/                           # Self-improving optimization loops
├── optimize_zinc.ts             #   ZINC loop: rsync → build → run → agent → keep/revert
├── optimize_zinc.test.ts        #   Tests for ZINC loop
├── implement_metal.ts           #   Metal implementation loop
└── implement_metal.test.ts      #   Tests for Metal loop

docs/                            # Technical documentation (published to site)
├── DEVELOPMENT.md               #   Development guide (canonical dev reference)
├── GETTING_STARTED.md           #   First run guide
├── RUNNING_ZINC.md              #   CLI usage and server mode
├── API.md                       #   OpenAI-compatible API spec
├── SPEC.md                      #   Architecture overview
├── RDNA4_TUNING.md              #   RDNA4-specific optimizations
├── GPU_REFERENCE.md             #   RDNA3/RDNA4 hardware reference
├── TURBOQUANT_SPEC.md           #   TurboQuant KV cache compression spec
├── APPLE_SILICON_REFERENCE.md   #   Apple Silicon M1–M5 reference
├── APPLE_METAL_REFERENCE.md     #   Metal/MSL kernel reference
└── APPLE_SILICON_METAL_ENABLEMENT.md # Metal port implementation notes

site/                            # Astro website (zolotukhin.ai)
tools/                           # API benchmark, standalone utilities
specs/                           # Feature specs and planning artifacts
scripts/                         # Deployment scripts
tests/                           # TypeScript test files
```

## Module Dependency Graph

```
main.zig
├── vulkan/instance.zig
├── vulkan/gpu_detect.zig
├── model/loader.zig
│   ├── model/gguf.zig
│   ├── vulkan/buffer.zig
│   └── vulkan/command.zig
├── model/tokenizer.zig
└── compute/forward.zig
    ├── compute/graph.zig
    ├── compute/dmmv.zig        → vulkan/pipeline.zig → shaders/dmmv_*.spv
    ├── compute/elementwise.zig → vulkan/pipeline.zig → shaders/{rms_norm,swiglu,rope}.spv
    ├── compute/attention.zig   → vulkan/pipeline.zig → shaders/flash_attn.spv
    └── model/architecture.zig  → compute/graph.zig
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
- Commit `.env`, credentials, or private IPs/ports
- Modify `.spv` binaries directly — always recompile from `.comp` source
- Add runtime dependencies beyond Vulkan and system libc
- Use wave32 without benchmarking against wave64 first

## Remote Test Node

An RDNA4 test node (AMD Radeon AI PRO R9700, 32GB, 576 GB/s) is available via SSH. Credentials are in `.env` (gitignored) as `ZINC_HOST`, `ZINC_USER`, `ZINC_PORT`.

## Running Benchmarks

### Baseline: 107 tok/s (2026-03-26)

The reference baseline is llama.cpp server on the RDNA4 test node with this exact configuration. All ZINC numbers are compared against this.

**Model**: `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` (20.7 GiB, MoE 35B/3B active)
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
#    -ngl 99 --device Vulkan0 --parallel 4 -c 32768
#    -ctk q8_0 -ctv q8_0 -b 4096 -ub 1024 --mlock --flash-attn on
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

### Measure ZINC

```bash
source .env

# Sync source to test node
rsync -az --delete --exclude '.zig-cache' --exclude 'zig-out' --exclude 'node_modules' \
  --exclude '.DS_Store' --exclude 'site' \
  -e "ssh -p $ZINC_PORT" . $ZINC_USER@$ZINC_HOST:/root/zinc/

# Build and run
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "cd /root/zinc && zig build -Doptimize=ReleaseFast && \
  RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
  -m /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt 'The capital of France is'"

# Key output lines:
#   info(forward): Prefill complete: N tokens in X ms (Y tok/s)
#   info(forward): Generated N tokens in X ms — Y tok/s (Z ms/tok)
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
    -m /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
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

Latest single-stream reference results with `zig build -Doptimize=ReleaseFast`:

**AMD RDNA4** (Radeon AI PRO R9700, 32 GB, 2026-03-31):
- CLI plain decode on `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`: `37.95 tok/s`, `26.3 ms/tok`
- CLI plain decode on `Qwen3.5-2B-Q4_K_M.gguf`: `26.71 tok/s`, `37.4 ms/tok`

**Apple Silicon** (M1 Max 32 GB, 2026-04-02):
- CLI plain decode on `Qwen3.5-2B-Q4_K_M.gguf`: `~17 tok/s`
- CLI plain decode on `Qwen3-8B-Q4_K_M.gguf`: `~8 tok/s`

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
    --model /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    --iterations 200 --warmup 25"
```

Focused single-case runs:

```bash
source .env

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && \
  zig build hot-bench -Doptimize=ReleaseFast -- \
    --model /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    --case q8_router"

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && \
  zig build hot-bench -Doptimize=ReleaseFast -- \
    --model /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    --case q8_shared_gate_up"

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && \
  RADV_DEBUG=shaderstats zig build hot-bench -Doptimize=ReleaseFast -- \
    --model /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
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

## Code Reference

Quick lookup for key types, functions, and their locations. Use this to navigate directly instead of searching.

### forward.zig — Inference Runtime (~2000 lines)

The main file. Contains the decode loop, all layer dispatch, MoE routing, SSM state.

| Symbol | Line | What it does |
|--------|------|-------------|
| `DecodeState` | ~26 | Per-request state: position counter, generated tokens list |
| `dequantRow()` | ~69 | CPU dequant: F32/F16/Q8_0/Q6_K/Q5_K/Q4_K row → f32 buffer |
| `topKSoftmax()` | ~280 | CPU MoE routing: softmax all experts → pick top-k → renormalize |
| `InferenceEngine` | ~200 | **Core runtime object** — owns all GPU buffers, dispatchers, KV cache, SSM state |
| `InferenceEngine.init()` | ~250 | Allocates everything: intermediate buffers, KV cache (40 layers), SSM state, descriptor pools |
| `InferenceEngine.decodeStep()` | ~430 | **One token**: embed → 40 layers (attn/SSM + MoE FFN) → final norm → LM head → logits readback |
| `InferenceEngine.prefillBatch()` | ~555 | Batch all prompt tokens in single GPU submission |
| `InferenceEngine.sampleGreedy()` | ~700 | CPU argmax over mapped logits staging buffer |
| `generate()` | ~750 | **Top-level**: prefill → decode loop → return token IDs |

**Decode step per layer** (inside `decodeStep`):
1. `attn_norm` — RMS norm of hidden state
2. **Attention layers** (every 4th layer): QKV projection → deinterleave Q+gate → sigmoid gate → RoPE → KV cache write → flash attention → output projection → residual add
3. **SSM layers** (other layers): QKV projection → conv1d → delta-net state update → gated RMS norm → output projection → residual add
4. **FFN (all layers)**: ffn_norm → MoE gate (GPU) → readback router logits (CPU) → topKSoftmax → dispatch top-k experts (gate+up → SwiGLU → down) → accumulate → shared expert → residual add

**Key buffers**: `hidden_buf`, `residual_buf`, `norm_buf`, `q_buf`, `k_buf`, `v_buf`, `attn_out_buf`, `gate_buf`, `up_buf`, `swiglu_buf`, `down_buf`, `moe_out_buf`, `router_logits_buf`, `router_staging`, `logits_buf`, `logits_staging`, `embed_staging`

**KV cache**: `kv_k_cache[layer]`, `kv_v_cache[layer]` — flat F32, [context_length × kv_dim]

**SSM state**: `ssm_conv_states[layer]` (CPU f32), `ssm_states[layer]` (CPU f32) — updated on CPU each token

### dmmv.zig — Matrix-Vector Dispatch

| Symbol | What it does |
|--------|-------------|
| `DmmvPushConstants` | Push constants: M (rows), K (cols), a_offset, x_offset, y_offset |
| `DmmvDispatch.init()` | Load pipelines for Q4_K/Q5_K/Q6_K/Q8_0/F16/F32; sets SPEC_K=hidden_dim |
| `DmmvDispatch.recordDispatch()` | Bind descriptor set + push constants, dispatch (M+63)/64 workgroups |
| `DmmvDispatch.pipelineForType()` | Select pipeline by GGMLType enum |

### elementwise.zig — Fused Ops

| Function | Push Constants | Workgroups |
|----------|---------------|------------|
| `recordRmsNorm()` | hidden_dim, n_tokens, eps | 1 per token |
| `recordSwiglu()` | n_elements | (n+63)/64 |
| `recordRope()` | stride, rope_dim, n_heads, position, freq_base | 1 per head |
| `recordDeinterleave()` | head_dim, n_heads | (n_heads+63)/64 |
| `recordSigmoidMul()` | n_elements | (n+63)/64 |
| `recordVadd()` | n_elements | (n+63)/64 |
| `recordScaleAcc()` | n_elements, scale | (n+63)/64 |
| `recordSsmConv1d()` | d_inner, d_conv | (d_inner+63)/64 — 1D causal conv for SSM layers |
| `recordSsmDeltaNet()` | d_inner, d_state, n_heads | 1 per head — delta-net recurrent state update |
| `recordSsmGatedNorm()` | n_elements, eps | (n+63)/64 — gated RMS norm for SSM output |
| `recordSoftmaxTopk()` | n_experts, k | 1 WG — softmax + top-k expert selection |
| `recordSigmoidScaleAcc()` | n_elements | (n+63)/64 — a[i] += sigmoid(gate[i]) * b[i] (shared expert gating) |

### attention.zig — Flash Attention

| Symbol | What it does |
|--------|-------------|
| `FlashAttnPush` | head_dim, n_heads, n_kv_heads, seq_len, page_size |
| `recordFlashAttn()` | 1 workgroup per query head, 256-token blocks, online softmax |

### graph.zig — Compute Graph IR

| Symbol | What it does |
|--------|-------------|
| `OpType` | 28 operation types (dmmv, flash_attn, rms_norm_mul, swiglu, rope, moe_gate, etc.) |
| `Graph.addNode()` | Add operation node, returns ID |
| `Graph.addDependency()` | Wire node ordering |
| `Graph.topologicalOrder()` | Sort for execution |
| `Graph.writeJsonReport()` | Export for analysis |

### gguf.zig — GGUF Parser

| Symbol | What it does |
|--------|-------------|
| `GGMLType` | Enum: f32, f16, q4_0..q8_k (31 formats) |
| `GGMLType.blockSize()` | Elements per block (1 for scalar, 256 for K-quants) |
| `GGMLType.bytesPerBlock()` | Bytes per block (e.g. Q4_K=144, Q8_0=34, F16=2) |
| `GGUFFile.getString/getU32/getF32()` | Metadata access |
| `GGUFFile.findTensor(name)` | Lookup tensor by name → TensorInfo (dims, type, offset) |

### loader.zig — Model Loading

| Symbol | What it does |
|--------|-------------|
| `ModelConfig` | All model dimensions: n_layers, n_heads, n_kv_heads, head_dim, hidden_dim, vocab_size, rope_dim, n_experts, SSM params |
| `Model` | Config + loaded tensors + mmap data |
| `load(path, instance, cmd_pool, allocator)` | Parse GGUF → extract config → mmap file → upload tensors to GPU |

### architecture.zig — Graph Builders

| Function | What it builds |
|----------|---------------|
| `buildDecodeGraph()` | Dispatch to arch-specific builder |
| `buildLlamaDecodeGraph()` | Standard transformer: norm → QKV → RoPE → attn → FFN |
| `buildMoeDecodeGraph()` | Transformer + MoE FFN with expert routing |
| `buildMambaDecodeGraph()` | Hybrid: interleave SSM + full-attention layers |

### tokenizer.zig — BPE Tokenizer

| Symbol | What it does |
|--------|-------------|
| `Tokenizer.initFromGGUF()` | Load vocab + merges from GGUF metadata |
| `Tokenizer.encode(text)` | UTF-8 → GPT-2 byte-level → BPE merges → token IDs |
| `gpt2ByteToUnicode()` | Byte-to-unicode mapping (printable=self, others=U+0100+) |

### Vulkan Layer

| File | Key Type | What it does |
|------|----------|-------------|
| `instance.zig` | `Instance` | Vulkan init, device selection, queue families |
| `buffer.zig` | `Buffer` | GPU memory: `initDeviceLocal()`, `initStaging()`, `upload()` |
| `pipeline.zig` | `Pipeline` | SPIR-V → compute pipeline with descriptor layout |
| `command.zig` | `CommandBuffer` | `begin()`, `dispatchWithPush()`, `computeBarrier()`, `submitAndWait()` |
| `gpu_detect.zig` | `GpuConfig` | Vendor classify → tuning params (wave_size, bandwidth_gbps, tile sizes) |

### Shaders

| Shader | Format | Workgroup | Notes |
|--------|--------|-----------|-------|
| `dmmv_q4k.comp` | Q4_K | 64 threads, 1 row/thread | Shared memory for input vector, SPEC_K specialization |
| `dmmv_q5k.comp` | Q5_K | 64 threads | Interleaved element layout (2e, 2e+1) |
| `dmmv_q6k.comp` | Q6_K | 64 threads | 210 bytes/block |
| `dmmv_q8_0.comp` | Q8_0 | 64 threads, 2 rows/WG | Simpler layout, good cache behavior |
| `dmmv_f16.comp` | F16 | 64 threads, 2 rows/WG | Direct multiply, no dequant |
| `dmmv_f32.comp` | F32 | 64 threads | Baseline, no quantization |
| `flash_attn.comp` | — | 1 WG/head | Paged KV, 256-token blocks, online softmax, GQA |
| `rms_norm_mul.comp` | — | 1 WG/token | y = weight * (x / sqrt(mean(x²) + eps)) |
| `swiglu.comp` | — | 64 threads | y = silu(gate) * up |
| `rope_fused.comp` | — | 1 WG/head | Partial rotation (IMRoPE): rotate rope_dim dims, copy rest |
| `deinterleave.comp` | — | 64 threads | Split [Q,gate] interleaved → separate Q, gate buffers |
| `sigmoid_mul.comp` | — | 64 threads | y = sigmoid(gate) * x (SSM gating) |
| `vadd.comp` | — | 64 threads | c = a + b |
| `scale_accumulate.comp` | — | 64 threads | a[i] += scale * b[i] (MoE expert accumulation) |
| `softmax_topk.comp` | — | 64 threads | Expert routing (GPU-side, used by recordSoftmaxTopk) |
| `ssm_conv1d.comp` | — | 64 threads | 1D causal convolution for SSM layers (d_conv=4 window) |
| `ssm_delta_net.comp` | — | 1 WG/head | Delta-net recurrent state update (decay + gate + input) |
| `ssm_gated_norm.comp` | — | 64 threads | Gated RMS norm: out = gate * rms_norm(x) |
| `scale_acc_sigmoid.comp` | — | 64 threads | a[i] += sigmoid(gate[i]) * b[i] (shared expert gating) |
| `sigmoid_scale_acc.comp` | — | 64 threads | Variant: sigmoid-gated scale-accumulate |
| `argmax.comp` | — | 64 threads | GPU-side argmax over logits |
| `embed_dequant_q4k.comp` | Q4_K | 64 threads | Token embedding dequant on GPU (future) |
| `coop_matmul.comp` | — | 16×16 | Cooperative matrix for batched prefill (future) |

### Critical Constants

| Constant | Value | Location |
|----------|-------|----------|
| KV cache max context | 4096 tokens | forward.zig |
| Push constant limit | 64 bytes | graph.zig |
| Page size (flash attn) | 16 tokens | forward.zig |
| Default workgroup size | 64 threads | all shaders |
| SPEC_K | hidden_dim (2048) | dmmv.zig init |
| Agent timeout | 30 min | optimize_zinc.ts:606 |
| Keep threshold | max(+3 tok/s, +2%) | optimize_zinc.ts |
| GPU lock file | /tmp/zinc-gpu.lock | optimize_zinc.ts |

### Decode Loop Data Flow

```
Token ID
  → CPU dequant embedding (token_embd.weight)
  → Upload to hidden_buf via staging
  → For each layer 0..39:
      ├─ RMS norm (hidden → norm_buf)
      ├─ QKV projection (DMMV: attn_q.weight × norm_buf → q/k/v)
      ├─ IF attention layer (every 4th):
      │   ├─ Deinterleave Q+gate → separate buffers
      │   ├─ Sigmoid gate × Q
      │   ├─ RoPE on Q and K (partial, rope_dim=64 of head_dim=256)
      │   ├─ Write K,V to KV cache at position
      │   ├─ Flash attention (Q × cached K/V → attn_out)
      │   └─ Output projection (DMMV: o_proj × attn_out)
      ├─ ELSE SSM layer:
      │   ├─ Conv1d (CPU, sliding window)
      │   ├─ Delta-net state update (CPU, d_state=128)
      │   ├─ Gated RMS norm (GPU)
      │   └─ Output projection (DMMV: ssm_out × ssm_hidden)
      ├─ Residual add (hidden += layer_output)
      ├─ FFN norm (RMS norm)
      ├─ MoE routing:
      │   ├─ Gate projection (DMMV: gate_exps × ffn_norm → router_logits)
      │   ├─ GPU→CPU readback of router logits
      │   ├─ CPU topKSoftmax → 8 expert IDs + weights
      │   └─ For each expert: gate+up (DMMV) → SwiGLU → down (DMMV) → scale_accumulate
      ├─ Shared expert: same as above but single expert, added to MoE output
      └─ Residual add (hidden += ffn_output)
  → Final RMS norm
  → LM head projection (DMMV: output.weight × norm → logits)
  → GPU→CPU readback logits
  → CPU argmax → next token ID
```
