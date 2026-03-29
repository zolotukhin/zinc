# ZINC Serving API Reference

ZINC exposes an OpenAI-compatible HTTP API. Point any OpenAI SDK client at ZINC by changing the base URL — no code changes required.

```bash
# Start the server
./zig-out/bin/zinc -m /path/to/model.gguf -p 8080

# Use with any OpenAI client
export OPENAI_BASE_URL=http://localhost:8080/v1
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat inference (streaming and non-streaming) |
| POST | `/v1/completions` | Text completion |
| GET | `/v1/models` | List loaded models |
| GET | `/health` | Server health, GPU stats, inference metrics |

---

## POST /v1/chat/completions

Generate a chat completion from a conversation. Supports both streaming (SSE) and non-streaming responses.

### Request

```json
{
  "model": "qwen",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": true,
  "stop": ["\n\n"]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | required | Model identifier (matches loaded model name or any string) |
| `messages` | array | required | Conversation messages with `role` and `content` |
| `max_tokens` | integer | 256 | Maximum tokens to generate |
| `temperature` | float | 1.0 | Sampling temperature. `0` = greedy (deterministic) |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `stream` | boolean | false | Enable Server-Sent Events streaming |
| `stop` | string or array | null | Stop sequence(s) to halt generation |

#### Message roles

| Role | Description |
|------|-------------|
| `system` | System prompt — sets behavior and context |
| `user` | User message |
| `assistant` | Previous assistant response (for multi-turn) |

### Response (non-streaming)

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1711500000,
  "model": "Qwen3.5-35B-A3B-Q4_K",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 8,
    "total_tokens": 32
  }
}
```

### Response (streaming)

When `stream: true`, the server responds with `Content-Type: text/event-stream`. Each event is a JSON object:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1711500000,"model":"Qwen3.5-35B-A3B-Q4_K","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1711500000,"model":"Qwen3.5-35B-A3B-Q4_K","choices":[{"index":0,"delta":{"content":" capital"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1711500000,"model":"Qwen3.5-35B-A3B-Q4_K","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Example: curl

```bash
# Non-streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128
  }'

# Streaming
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128,
    "stream": true
  }'
```

### Example: Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

# Non-streaming
response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=128,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Explain gravity"}],
    max_tokens=256,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Example: Node.js (OpenAI SDK)

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "not-needed",
});

// Non-streaming
const response = await client.chat.completions.create({
  model: "qwen",
  messages: [{ role: "user", content: "What is 2+2?" }],
  max_tokens: 128,
});
console.log(response.choices[0].message.content);

// Streaming
const stream = await client.chat.completions.create({
  model: "qwen",
  messages: [{ role: "user", content: "Explain gravity" }],
  max_tokens: 256,
  stream: true,
});
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || "");
}
```

---

## POST /v1/completions

Generate a text completion from a raw prompt string. Same parameters as chat completions, but uses `prompt` instead of `messages`.

### Request

```json
{
  "model": "qwen",
  "prompt": "The capital of France is",
  "max_tokens": 64,
  "temperature": 0.0,
  "stream": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | required | Model identifier |
| `prompt` | string | required | Raw text prompt |
| `max_tokens` | integer | 256 | Maximum tokens to generate |
| `temperature` | float | 1.0 | Sampling temperature |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `stream` | boolean | false | Enable SSE streaming |
| `stop` | string or array | null | Stop sequence(s) |

### Response

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1711500000,
  "model": "Qwen3.5-35B-A3B-Q4_K",
  "choices": [
    {
      "index": 0,
      "text": " Paris, the largest city in France.",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 9,
    "total_tokens": 15
  }
}
```

---

## GET /v1/models

List all currently loaded models. ZINC loads one model at a time.

### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3.5-35B-A3B-Q4_K",
      "object": "model",
      "created": 1711500000,
      "owned_by": "zinc"
    }
  ]
}
```

---

## GET /health

Server health check for monitoring and load balancers. Returns 200 when ready, 503 when the model is still loading.

### Response (ready)

```json
{
  "status": "ok",
  "gpu": {
    "name": "AMD Radeon Graphics (RADV GFX1201)",
    "vendor": "amd_rdna4",
    "vram_total_mb": 32624,
    "vram_used_mb": 21504,
    "bandwidth_gbps": 576,
    "compute_units": 64
  },
  "model": {
    "name": "Qwen3.5-35B-A3B-Q4_K",
    "architecture": "qwen35moe",
    "parameters": "34.66B",
    "layers": 40,
    "context_length": 32768,
    "quantization": "Q4_K"
  },
  "inference": {
    "active_requests": 2,
    "max_parallel": 4,
    "tokens_generated": 142857,
    "avg_decode_tps": 108.5,
    "uptime_seconds": 3600
  }
}
```

### Response (loading)

```
HTTP/1.1 503 Service Unavailable

{"status": "loading"}
```

---

## Error responses

All errors follow the OpenAI error schema:

```json
{
  "error": {
    "message": "Invalid request: missing 'messages' field",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

| Status | Type | When |
|--------|------|------|
| 400 | `invalid_request_error` | Malformed JSON, missing required fields, invalid parameters |
| 404 | `model_not_found` | Requested model doesn't match loaded model |
| 429 | `rate_limit_exceeded` | All request slots are occupied |
| 500 | `internal_error` | GPU error, inference failure |
| 503 | `service_unavailable` | Model still loading, or KV cache pages exhausted |

---

## CORS

All endpoints include `Access-Control-Allow-Origin: *` for browser-based clients.

## Connection handling

- Streaming responses use HTTP/1.1 chunked transfer encoding
- Client disconnection during streaming immediately stops token generation and frees KV cache pages
- Connections are closed after each response (`Connection: close`)

## Concurrency

ZINC supports up to `--parallel N` concurrent requests (default: 4). Each request gets its own:
- Decode state and token buffer
- KV cache page allocation
- Independent sampling state

Requests are batched by the continuous batching scheduler — prefill and decode steps are interleaved across active requests in a single GPU submission. There is no per-slot throughput degradation at the designed concurrency level.

## Server configuration

```
./zig-out/bin/zinc [options]
  -m, --model <path>       Path to GGUF model file (required)
  -p, --port <port>        Server port (default: 8080)
  -d, --device <id>        Vulkan device index (default: 0)
  -c, --context <size>     Context length (default: 4096)
  --parallel <n>           Max concurrent requests (default: 4)
  --kv-quant <bits>        TurboQuant KV cache compression: 0/2/3/4 (default: 0=off)
  --prompt <text>          Single prompt mode (no server, CLI only)
```

## Supported models

ZINC loads models in GGUF format with the following quantization types:

| Format | Supported | Notes |
|--------|-----------|-------|
| Q4_K | Yes | Primary target, hand-tuned DMMV shader |
| Q5_K | Yes | Hand-tuned DMMV shader |
| Q6_K | Yes | Hand-tuned DMMV shader |
| Q8_0 | Yes | Used for attention weights |
| F16 | Yes | For KV cache and small tensors |
| F32 | Yes | Baseline, no quantization overhead |

Supported architectures: LLaMA, Mistral, Qwen2, Qwen2-MoE (including Qwen3.5 hybrid SSM+attention), Mamba, Jamba.
