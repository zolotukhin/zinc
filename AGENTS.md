# AGENTS.md — ZINC Inference Engine

Instructions for AI coding agents working on this repository.

## Commands

```bash
# Build (shaders compile on Linux only; macOS skips GPU inference)
zig build

# Run inference
./zig-out/bin/zinc -m model.gguf --prompt "Hello" [-d device_id] [--kv-quant 3]

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
├── main.zig                    # CLI entry point, arg parsing, main loop
├── vulkan/                     # GPU abstraction layer
│   ├── vk.zig                  #   @cImport of vulkan/vulkan.h
│   ├── instance.zig            #   VkInstance, VkDevice, queue init
│   ├── buffer.zig              #   GPU buffer alloc, staging, DMA
│   ├── pipeline.zig            #   Compute pipeline from SPIR-V
│   ├── command.zig             #   Command pool/buffer, dispatch recording
│   └── gpu_detect.zig          #   GPU vendor detect + auto-tuning (RDNA3/4, NVIDIA, Intel)
├── model/                      # Model loading
│   ├── gguf.zig                #   GGUF format parser (header, metadata, tensors)
│   ├── loader.zig              #   Load GGUF → GPU buffers (mmap + DMA)
│   ├── architecture.zig        #   Compute graph builders per arch (Llama, Qwen, Mamba)
│   └── tokenizer.zig           #   Text ↔ tokens (shells to Python, native Zig planned)
├── compute/                    # Inference dispatch
│   ├── graph.zig               #   Static compute graph IR (nodes, deps, topo sort)
│   ├── dmmv.zig                #   Decode matmul-vec dispatch (Q4_K, Q8_0, F16)
│   ├── elementwise.zig         #   RMS norm, SwiGLU, RoPE dispatch
│   ├── attention.zig           #   Flash attention dispatch (paged, GQA)
│   └── forward.zig             #   Inference engine: decode loop, token generation
└── shaders/                    # GLSL compute kernels → compiled to .spv
    ├── dmmv_q4k.comp           #   Q4_K dequant + matvec
    ├── dmmv_q8_0.comp          #   Q8_0 dequant + matvec
    ├── dmmv_f16.comp           #   FP16 matvec (no dequant)
    ├── coop_matmul.comp        #   Cooperative matrix 16x16x16 for prefill
    ├── flash_attn.comp         #   Paged flash attention with GQA
    ├── rms_norm_mul.comp       #   RMS norm + scale multiply (fused)
    ├── swiglu.comp             #   SiLU-gated linear unit (fused)
    ├── rope_fused.comp         #   Rotary position embedding (fused)
    ├── sigmoid_mul.comp        #   Sigmoid * element-wise (SSM gating)
    ├── softmax_topk.comp       #   Softmax + top-k (MoE routing)
    ├── tq_quantize_keys.comp   #   TurboQuant: quantize K cache
    ├── tq_quantize_values.comp #   TurboQuant: quantize V cache
    ├── tq_attention_scores.comp#   TurboQuant: attention on quantized KV
    └── tq_decompress_values.comp#  TurboQuant: decompress V values

benchmarks/
├── bandwidth.zig               # DMMV bandwidth utilization benchmark
└── dispatch.zig                # Vulkan dispatch overhead benchmark

loops/                          # Self-improving optimization loops
├── optimize_zinc.ts            #   ZINC loop: rsync → build → run → agent → keep/revert
├── optimize_zinc.test.ts       #   Tests for ZINC loop (bun test)
├── optimize_llm_tps.ts         #   llama.cpp TPS optimization loop
└── optimize_llm_tps.test.ts    #   Tests for llama.cpp loop (bun test)

docs/                           # Technical specifications
├── SPEC.md                     #   Architecture overview
├── API.md                      #   OpenAI-compatible API spec
├── RDNA4_TUNING.md             #   RDNA4-specific optimizations
└── TURBOQUANT_SPEC.md          #   TurboQuant KV cache compression spec

site/                           # Astro website (zolotukhin.ai)
specs/                          # Feature specs and planning artifacts
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
    curl -s http://localhost:8088/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"q\",\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}],\"max_tokens\":256,\"stream\":false}" \
      | python3 -c "import sys,json; d=json.load(sys.stdin); t=d.get(\"timings\",{}); print(f\"Run {i}: gen {t.get(\"predicted_per_second\",0):.1f} tok/s | prompt {t.get(\"prompt_per_second\",0):.1f} tok/s\")"
  done
'
# Expected: ~107 tok/s generation, ~220 tok/s prompt (runs 2-3, after warmup)
```

### Measure ZINC

```bash
source .env

# Sync source to test node
rsync -az --delete --exclude '.zig-cache' --exclude 'zig-out' --exclude 'node_modules' \
  --exclude '.DS_Store' --exclude 'site' --exclude 'research/turboquant-pytorch-master' \
  -e "ssh -p $ZINC_PORT" . $ZINC_USER@$ZINC_HOST:/root/zinc/

# Build and run
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "cd /root/zinc && zig build && \
  RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
  -m /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt 'The capital of France is'"

# Key output lines:
#   info(forward): Prefill complete: N tokens in X ms (Y tok/s)
#   info(forward): Generated N tokens in X ms — Y tok/s (Z ms/tok)
```

### Troubleshooting performance

If llama.cpp baseline drops below ~100 tok/s, check in order:
1. **Mesa version** — `dpkg -l mesa-vulkan-drivers` must show 25.0.7 (not 25.2.8)
2. **GECC** — `cat /sys/module/amdgpu/parameters/ras_enable` must show 0
3. **coop_matrix** — server log must show `matrix cores: KHR_coopmat`
4. **Reboot** — Mesa/driver changes need a reboot to take full effect

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
| `softmax_topk.comp` | — | 64 threads | Expert routing (not currently used — CPU routing) |
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
