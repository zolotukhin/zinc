# ZINC API Reference

## OpenAI-Compatible Endpoints

### POST /v1/chat/completions

Chat inference with streaming support.

```bash
curl http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": true
  }'
```

### POST /v1/completions

Text completion.

### POST /v1/embeddings

Embedding extraction.

### GET /v1/models

List loaded models.

### GET /health

Server health and GPU statistics.

```json
{
  "status": "ok",
  "gpu": {
    "name": "AMD Radeon Graphics (RADV GFX1201)",
    "vram_total_mb": 32768,
    "vram_used_mb": 21504,
    "temperature_c": 42,
    "clock_mhz": 2350
  },
  "model": {
    "name": "Qwen3.5-35B-A3B-Q4_K",
    "parameters": "34.66B",
    "context_length": 32768
  },
  "inference": {
    "active_requests": 3,
    "tokens_generated": 142857,
    "avg_generation_tps": 108.5
  }
}
```
