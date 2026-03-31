# ZINC Serving API Reference

ZINC exposes an OpenAI-compatible HTTP API. Point any OpenAI SDK client at ZINC by changing the base URL — no code changes required.

```bash
# Start the server (append --debug or use ZINC_DEBUG=1 for diagnostic logs)
./zig-out/bin/zinc -m /path/to/model.gguf -p 8080

# Use with any OpenAI client
export OPENAI_BASE_URL=http://localhost:8080/v1
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat inference (streaming and non-streaming) |
| POST | `/v1/completions` | Text completion |
| GET | `/v1/models` | List managed models, fit status, install state, and the active entry |
| POST | `/v1/models/activate` | Activate an installed managed model |
| GET | `/health` | Server health, active requests, queued requests, and uptime |

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
  "enable_thinking": true,
  "top_p": 0.9,
  "stream": true,
  "stop": ["\n\n"]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | required | OpenAI-compatibility field. Generation currently runs on the server's active model. |
| `messages` | array | required | Conversation messages with `role` and `content` |
| `max_tokens` | integer | 256 | Maximum tokens to generate |
| `temperature` | float | 1.0 | Sampling temperature. `0` = greedy (deterministic) |
| `enable_thinking` | boolean | false | When the active model's chat template supports it, request an open `<think>` block instead of the no-thinking generation suffix |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `stream` | boolean | false | Enable Server-Sent Events streaming |
| `stop` | string or array | null | Stop sequence(s) to halt generation |

`enable_thinking` is currently model-dependent. Qwen-style templates that expose `enable_thinking` and `<think>` honor it; models without that template support ignore the flag.

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
    "max_tokens": 128,
    "enable_thinking": true
  }'

# Streaming
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128,
    "enable_thinking": true,
    "stream": true
  }'
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
  enable_thinking: true,
});
console.log(response.choices[0].message.content);

// Streaming
const stream = await client.chat.completions.create({
  model: "qwen",
  messages: [{ role: "user", content: "Explain gravity" }],
  max_tokens: 256,
  enable_thinking: true,
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

List managed models, fit status, install state, and the active entry. ZINC still loads one model into memory at a time, but this endpoint exposes the built-in managed catalog for the current server GPU profile.

### Response

```json
{
  "object": "list",
  "profile": "amd-rdna4-32gb",
  "data": [
    {
      "id": "qwen35-35b-a3b-q4k-xl",
      "object": "model",
      "created": 1711500000,
      "owned_by": "zinc",
      "display_name": "Qwen3.5 35B-A3B UD Q4_K_XL",
      "homepage_url": "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF",
      "installed": false,
      "active": false,
      "managed": true,
      "supported_on_current_gpu": true,
      "fits_current_gpu": true,
      "required_vram_bytes": 22987514102,
      "fit_source": "catalog",
      "status": "supported"
    }
  ]
}
```

---

## POST /v1/models/activate

Activate an installed managed model in a running server. The model must already exist in the local managed cache and must fit the current GPU budget.

### Request

```json
{
  "model": "qwen35-2b-q4k-m"
}
```

### Response

```json
{
  "object": "model.activation",
  "id": "qwen35-2b-q4k-m",
  "active": true
}
```

### Example: curl

```bash
# Switch the running server to an installed managed model
curl http://localhost:8080/v1/models/activate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35-2b-q4k-m"
  }'
```

---

## GET /health

Server health check for monitoring and load balancers.

### Response

```json
{
  "status": "ok",
  "model": "qwen3.5-35b",
  "active_requests": 1,
  "queued_requests": 0,
  "uptime_seconds": 3600
}
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
| 400 | `invalid_request_error` | Unknown managed model id, model not installed, or model does not fit when activating |
| 500 | `internal_error` | GPU error, inference failure |

---

## CORS

All endpoints include `Access-Control-Allow-Origin: *` for browser-based clients.

## Connection handling

- Streaming responses use HTTP/1.1 chunked transfer encoding
- Client disconnection during streaming immediately stops token generation and frees KV cache pages
- Connections are closed after each response (`Connection: close`)

## Concurrency

ZINC accepts overlapping requests, but generation is currently serialized behind one engine lock. That means:

- one request generates at a time
- later requests queue behind the active generation
- `/health` reports `active_requests` and `queued_requests`
- the chat UI can switch models through `/v1/models/activate`, but the switch also takes the same generation lock

## Server configuration

```
./zig-out/bin/zinc [options]
  -m, --model <path>       Path to GGUF model file
  --model-id <id>         Managed model id from the built-in catalog/cache
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
