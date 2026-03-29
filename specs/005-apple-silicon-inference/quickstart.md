# Quickstart: Gather Apple Silicon Baselines

Run these on Mac Studio M4 Max 64 GB to establish performance targets.

## 1. llama.cpp Metal (uses same GGUF as Vulkan)

```bash
# Build
cd /tmp
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j16

# Download model (or copy from remote node)
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
  Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --local-dir ./models

# Benchmark (prefill 512 + generate 128)
./build/bin/llama-bench \
  -m ./models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  -ngl 99 -p 512 -n 128

# Interactive test
./build/bin/llama-cli \
  -m ./models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 4096 -p "The capital of France is" -n 256
```

## 2. mlx-lm (MLX native, 4-bit)

```bash
# Install
pip install -U mlx-lm

# Auto-downloads model from HuggingFace on first run (~19 GB)
mlx_lm.generate \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --prompt "The capital of France is" \
  --max-tokens 256 \
  --verbose

# Chat mode
mlx_lm.chat --model mlx-community/Qwen3.5-35B-A3B-4bit
```

## 3. vllm-mlx (parallel request benchmark)

```bash
# Install
pip install vllm-mlx

# Start server
vllm serve mlx-community/Qwen3.5-35B-A3B-4bit \
  --host 0.0.0.0 --port 8080

# Single request
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"q","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":256}'

# 5 concurrent requests (measure aggregate tok/s)
for i in $(seq 1 5); do
  curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"q","messages":[{"role":"user","content":"Write a haiku about request '$i'"}],"max_tokens":128}' &
done
wait
```

## What to Record

| Metric | llama.cpp | mlx-lm | vllm-mlx |
|--------|-----------|--------|----------|
| Model load time | | | |
| Prefill tok/s (512 tokens) | | | |
| Decode tok/s (single request) | | | |
| Decode tok/s (5 concurrent) | N/A | N/A | |
| Peak memory usage | | | |
| First token latency | | | |

Save results to `specs/005-apple-silicon-inference/baselines.md`.
