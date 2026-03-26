# API Contract: OpenAI-Compatible Endpoints

## POST /v1/chat/completions

**Request**:
```json
{
  "model": "string",
  "messages": [{"role": "system|user|assistant", "content": "string"}],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 1.0,
  "stream": false,
  "stop": ["string"]
}
```

**Response (non-streaming)**:
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "string",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "string"},
    "finish_reason": "stop|length"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
}
```

**Response (streaming, `stream: true`)**:
SSE events, one per token:
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"token"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## POST /v1/completions

**Request**:
```json
{
  "model": "string",
  "prompt": "string",
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": false
}
```

**Response**: Same structure as chat completions but with `"object": "text_completion"`.

## POST /v1/embeddings

**Request**:
```json
{
  "model": "string",
  "input": "string or [string]"
}
```

**Response**:
```json
{
  "object": "list",
  "data": [{"object": "embedding", "embedding": [0.1, 0.2, ...], "index": 0}],
  "model": "string",
  "usage": {"prompt_tokens": 10, "total_tokens": 10}
}
```

## GET /v1/models

**Response**:
```json
{
  "object": "list",
  "data": [{"id": "model-name", "object": "model", "created": 1234567890, "owned_by": "zinc"}]
}
```

## GET /health

**Response**:
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

## Error Responses

All endpoints return errors in OpenAI format:
```json
{
  "error": {
    "message": "string",
    "type": "invalid_request_error|server_error",
    "code": "string|null"
  }
}
```

HTTP status codes: 400 (bad request), 404 (model not found), 500 (server error), 503 (model loading).
