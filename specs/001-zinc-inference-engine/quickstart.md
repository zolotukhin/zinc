# Quickstart Validation: ZINC Inference Engine

## Prerequisites

- AMD RDNA3/RDNA4 GPU with Vulkan 1.3+ support
- RADV or AMDVLK driver installed
- `RADV_PERFTEST=coop_matrix` set in environment
- GPU ECC disabled: `amdgpu.ras_enable=0` in kernel cmdline (recommended)
- Zig 0.15.2+ installed
- System glslc (shaderc 2023.8 from Ubuntu 24.04 packages)
- A GGUF model file (e.g., Qwen3-8B-Q4_K.gguf)

## Validation Scenario 1: Single-Request Inference (US1)

```bash
# Build
zig build

# Run single inference
./zig-out/bin/zinc -m Qwen3-8B-Q4_K.gguf --prompt "The capital of France is"

# Expected: Coherent text generation at 120+ tok/s (RX 9070 XT) or 110+ tok/s (AI PRO R9700)
# Verify: Check stderr for bandwidth utilization stats (should show 67-93% on large matmuls)
```

**Pass criteria**:
- Text is coherent and contextually correct
- Generation speed meets target for the GPU
- No Vulkan validation errors
- Logit comparison against llama.cpp shows >99.5% cosine similarity

## Validation Scenario 2: Server + Concurrent Requests (US2)

```bash
# Start server
./zig-out/bin/zinc -m Qwen3-8B-Q4_K.gguf -p 8080

# In another terminal, send 4 concurrent requests
for i in 1 2 3 4; do
  curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen","messages":[{"role":"user","content":"Count to 100"}],"max_tokens":256,"stream":true}' &
done
wait

# Check health
curl http://localhost:8080/health
```

**Pass criteria**:
- All 4 streams complete without errors
- Each stream delivers 100+ tok/s
- Health endpoint shows correct GPU stats and active request count
- No memory leaks after requests complete (check health VRAM stats)

## Validation Scenario 3: TurboQuant KV Compression (US3)

```bash
# Run with TQ-3bit
./zig-out/bin/zinc -m Qwen3-8B-Q4_K.gguf -p 8080 --kv-quant 3

# Send 8 concurrent 8K-context requests (should fit in 16GB VRAM)
# Compare output quality against --kv-quant 0 (FP16)
```

**Pass criteria**:
- 8 concurrent requests run without OOM on RX 9070 XT (16GB)
- VRAM usage is ~5x lower for KV cache vs FP16
- Text quality is indistinguishable from FP16 in casual evaluation
- Attention cosine similarity >99.5% in benchmark mode

## Validation Scenario 4: MoE/Mamba Architecture (US4)

```bash
# Load Qwen3.5-35B-A3B (SSM+attention hybrid MoE)
./zig-out/bin/zinc -m Qwen3.5-35B-A3B-Q4_K.gguf --prompt "Explain quantum computing"

# Expected: Coherent output at 110+ tok/s on AI PRO R9700
```

**Pass criteria**:
- Expert routing selects correct top-k experts (validate via debug logging)
- SSM state-space operations produce correct outputs
- Output matches llama.cpp reference for same prompt and seed
