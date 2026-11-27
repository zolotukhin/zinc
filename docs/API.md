# ZINC Serving API Reference

ZINC exposes a local OpenAI-compatible HTTP API. Point OpenAI-style clients at
the server by changing the base URL.

```bash
# Start the server. Plain server mode defaults to port 8080.
./zig-out/bin/zinc --model-id qwen35-9b-q4k-m -p 8080

export OPENAI_BASE_URL=http://localhost:8080/v1
```

`zinc chat` is a convenience command for the built-in chat UI and defaults to
port 9090 unless `-p` is provided.

## Status & known gaps (read first)

Before you integrate, know these limits up front. Each is a deliberate state of the engine today, not a bug to file:

- **Decode is serialized.** Concurrent HTTP requests are accepted, but generation runs behind a single engine lock — request N+1 queues until request N finishes. Throughput scales with one stream, not N. Continuous batching is on the roadmap; until then, do not architect around parallel streams.
- **`/v1/embeddings` is not implemented.** ZINC is a chat/completion engine today; there is no embedding endpoint and no plan to fake one. If you need embeddings for RAG, run them in a separate process (llama.cpp's `llama-embedding`, a hosted API, etc.) and only point chat at ZINC.
- **`/v1/completions` is non-streaming and ignores sampling controls.** Use `/v1/chat/completions` for anything that needs streaming, `temperature`, `top_p`, or `enable_thinking`.
- **Client-supplied `stop` sequences are not honored.** Chat generation stops on the model/template end markers (`<|im_end|>`, etc.) only.
- **No `/v1/audio/*`, no image inputs, no fine-tuning endpoints, no batch API.** These are out of scope; if you need them, ZINC is not the right engine.
- **MTP depends on the loaded model and serving configuration.** CUDA/ROCm
  single-slot serving can use a compatible dense model's appended NextN block.
  `/health` and `/v1/models` report `supports_mtp`, `mtp_default_enabled`,
  `mtp_request_control`, and `mtp_unavailable_reason` from the initialized runtime.
  Multi-slot serving, unsupported models/backends, explicit `ZINC_MTP=0`, and
  setup failures report the reason instead of claiming acceleration. Missing
  fields from an older server leave the status unknown.

Everything else listed in the endpoint table below works as documented.

## Compatibility Notes

The API intentionally follows the OpenAI response shapes where practical, but
ZINC is still a local single-process engine:

- `/v1/chat/completions` supports streaming SSE, non-streaming responses,
  `temperature`, `top_p`, `enable_thinking`, optional `session_id` prompt reuse,
  and OpenAI-style tool calling for ChatML/Qwen-family templates.
- `/v1/completions` is currently a raw, non-streaming text-completion path.
  It accepts common compatibility fields, but sampling controls and `stream`
  are not applied on this endpoint yet.
- Request-provided `stop` sequences are not implemented yet. Chat generation
  still stops on model/template end markers such as `<|im_end|>`.
- Generation runs under one engine lock. Concurrent HTTP requests are accepted,
  but decode is serialized.
- The `model` request field is honored for managed models: if it names an
  installed catalog model different from the active model, ZINC attempts to
  activate it before generation.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat inference, streaming and non-streaming |
| POST | `/v1/completions` | Raw text completion, non-streaming |
| GET | `/v1/models` | Managed catalog, install state, active model, memory status |
| POST | `/v1/models/pull` | Download an installed-compatible managed model asynchronously |
| POST | `/v1/models/activate` | Activate an installed managed model |
| POST | `/v1/models/unload` | Release the loaded model from GPU memory while keeping cached files |
| POST | `/v1/models/remove` | Remove a cached managed model, optionally unloading it first |
| GET | `/health` | Health, queue counters, uptime, GPU memory/context status |
| GET | `/` or `/chat` | Built-in chat UI |

---

## POST /v1/chat/completions

Generate a chat completion from a message list.

### Request

```json
{
  "model": "qwen35-9b-q4k-m",
  "session_id": "chat-123",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "enable_thinking": false,
  "stream": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | current model | Managed model id to use. If installed and different from the active model, the server activates it before generation. |
| `session_id` | string | empty | Optional prompt-reuse key. Reusing a session id lets ZINC reuse the matching prefix for append-only chat histories. |
| `messages` | array | required | OpenAI-style messages. `content` may be a string, `null`, or an array of text content blocks. Non-text blocks are ignored. |
| `max_tokens` | integer | 256 | Maximum generated tokens before context-budget clamping. |
| `temperature` | float | 0.0 | Sampling temperature. `0` uses greedy decoding. Values are clamped to `0..2`. |
| `top_p` | float | 1.0 | Nucleus sampling threshold, clamped to `0..1`. |
| `enable_thinking` | boolean | model default | For templates that support it, request an open thinking block. Catalog entries with unstable thinking can force this off. |
| `enable_mtp` | boolean or null | server default | `true` requests model-native MTP; unsupported configurations return 400 before queueing. `false` uses ordinary decoding for this request. Omission or `null` preserves `mtp_default_enabled`. Only offer a request toggle when `mtp_request_control` is true. |
| `stream` | boolean | false | Enable Server-Sent Events streaming. |
| `tools` | array | empty | OpenAI function-tool definitions. Rendered for ChatML/Qwen-style templates when tool calling is enabled. |
| `tool_choice` | string or object | `auto` | `"none"` suppresses tool injection. `"auto"` and forced function-object choices are currently treated as automatic tool choice. |

Supported message roles:

| Role | Description |
|------|-------------|
| `system` | System prompt. |
| `user` | User message. |
| `assistant` | Previous assistant response. May include historical `tool_calls`. |
| `tool` | Tool result message with `tool_call_id`; rendered back into ChatML tool-response blocks for Qwen-style templates. |

### Tool Calling

Tool calling is enabled by default and can be disabled with:

```bash
ZINC_TOOL_CALLING=0 ./zig-out/bin/zinc --model-id qwen35-9b-q4k-m
```

ZINC never executes tools. It only renders tool definitions into the prompt,
parses model-emitted `<tool_call>...</tool_call>` blocks, returns structured
`tool_calls`, and accepts subsequent `role: "tool"` result messages from the
client.

Example request:

```json
{
  "model": "qwen35-9b-q4k-m",
  "messages": [{"role": "user", "content": "What time is it in UTC?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_time",
        "description": "Return the current time for a timezone.",
        "parameters": {
          "type": "object",
          "properties": {
            "timezone": {"type": "string"}
          },
          "required": ["timezone"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

Non-streaming tool-call response shape:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1711500000,
  "model": "Qwen 3.5 9B Q4_K_M",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_0",
            "type": "function",
            "function": {
              "name": "get_time",
              "arguments": "{\"timezone\":\"UTC\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 128,
    "completion_tokens": 12,
    "total_tokens": 140
  }
}
```

### Response

Non-streaming responses use the OpenAI chat-completion shape:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1711500000,
  "model": "Qwen 3.5 9B Q4_K_M",
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

When `stream: true`, the server responds with `Content-Type:
text/event-stream`. The first chunk includes the assistant role, content chunks
follow, and the final chunk carries `finish_reason`.

```text
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1711500000,"model":"Qwen 3.5 9B Q4_K_M","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1711500000,"model":"Qwen 3.5 9B Q4_K_M","choices":[{"index":0,"delta":{"content":"Paris"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1711500000,"model":"Qwen 3.5 9B Q4_K_M","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Streaming tool calls are emitted atomically as `delta.tool_calls` chunks, with
`finish_reason: "tool_calls"` in the final chunk when a tool call was produced.

### curl

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35-9b-q4k-m",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 64
  }'

curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35-9b-q4k-m",
    "messages": [{"role": "user", "content": "Explain gravity briefly."}],
    "max_tokens": 128,
    "stream": true
  }'
```

---

### MTP capability and execution status

Capability describes the initialized model/runtime, not a claim that a reply
used MTP. The chat UI uses `mtp_default_enabled` until the user chooses a mode;
unsupported or unreported capability never silently becomes an opt-in.

The final SSE chunk (before `[DONE]`) and non-streaming completion contain an
`mtp` object:

```json
{
  "mtp": {
    "requested": null,
    "enabled": true,
    "used": true,
    "drafted": 24,
    "accepted": 18,
    "cycles": 12,
    "fallback_reason": null
  }
}
```

`requested` preserves the nullable request option; `enabled` is the resolved
policy. `used` and the counters describe execution, so enabling MTP does not by
itself justify showing it as active. A request that falls back to ordinary
decoding reports `used: false` and a `fallback_reason`. No speedup is implied by
support, activation, or draft acceptance alone.

## POST /v1/completions

Generate from a raw text prompt without applying a chat template.

### Request

```json
{
  "model": "qwen35-9b-q4k-m",
  "prompt": "The capital of France is",
  "max_tokens": 64
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | current model | Managed model id to use; may activate an installed model before generation. |
| `prompt` | string | required | Raw text prompt. |
| `max_tokens` | integer | 256 | Maximum generated tokens before context-budget clamping. |
| `temperature` | float | accepted | Parsed for compatibility but not applied by this endpoint yet. |
| `stream` | boolean | accepted | Parsed for compatibility but ignored; responses are non-streaming JSON. |

### Response

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1711500000,
  "model": "Qwen 3.5 9B Q4_K_M",
  "choices": [
    {
      "index": 0,
      "text": " Paris.",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 2,
    "total_tokens": 8
  }
}
```

---

## GET /v1/models

List the managed catalog for the server's current GPU profile, including install
state, active model, fit decisions, download progress, and active GPU memory
usage. ZINC loads one model into memory at a time.

The top-level `supports_model_management`, `supports_model_unload`, and
`supports_model_pull` flags describe the available controls. The built-in UI
uses these capabilities rather than assuming every server can download or swap
models. CUDA/ROCm serving lists its startup model and installed catalog models;
it can load or unload them, while `supports_model_pull: false` means downloads
must be installed with the model CLI first.

### Response

```json
{
  "object": "list",
  "supports_model_management": true,
  "supports_model_unload": true,
  "supports_model_pull": true,
  "profile": "amd-rdna4-32gb",
  "active_memory_used_bytes": 22300000000,
  "active_memory_budget_bytes": 34359738368,
  "active_memory_weights_bytes": 21000000000,
  "active_memory_runtime_bytes": 1300000000,
  "active_context_reserved_bytes": 536870912,
  "active_context_active_bytes": 131072,
  "active_context_tokens": 128,
  "active_context_capacity_tokens": 4096,
  "data": [
    {
      "id": "qwen36-35b-a3b-q4k-xl",
      "object": "model",
      "created": 1711500000,
      "owned_by": "zinc",
      "context_length": 4096,
      "display_name": "Qwen3.6 35B-A3B UD Q4_K_XL",
      "release_date": "2026-04-15",
      "homepage_url": "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF",
      "family": "Qwen 3.6",
      "quantization": "Q4_K_XL",
      "size_bytes": 21000000000,
      "installed": true,
      "active": true,
      "managed": true,
      "supported_on_current_gpu": true,
      "fits_current_gpu": true,
      "required_vram_bytes": 23106019926,
      "required_vram_with_offload_bytes": 23106019926,
      "requires_offload_to_fit": false,
      "fit_source": "catalog",
      "status": "supported",
      "supports_thinking_toggle": false,
      "downloading": false,
      "download_phase": "idle",
      "downloaded_bytes": 0,
      "download_total_bytes": 0,
      "download_error": ""
    }
  ]
}
```

---

## POST /v1/models/pull

Download a supported managed model into the local cache. Downloads run in a
background thread; poll `GET /v1/models` for progress.

### Request

```json
{
  "model": "qwen35-9b-q4k-m"
}
```

### Responses

Already installed:

```json
{
  "object": "model.pull",
  "id": "qwen35-9b-q4k-m",
  "state": "installed"
}
```

Started download:

```json
{
  "object": "model.pull",
  "id": "qwen35-9b-q4k-m",
  "state": "downloading",
  "downloaded_bytes": 0,
  "download_total_bytes": 0
}
```

If the same model is already downloading, the server returns `202` with the
current `state`, `downloaded_bytes`, and `download_total_bytes`.

---

## POST /v1/models/activate

Activate an installed managed model in a running server. The model must exist in
the local managed cache and fit the current GPU budget.

### Request

```json
{
  "model": "qwen35-9b-q4k-m"
}
```

### Response

```json
{
  "object": "model.activation",
  "id": "qwen35-9b-q4k-m",
  "active": true
}
```

---

## POST /v1/models/unload

Release the current model, tokenizer, and inference buffers from GPU memory.
Downloaded model files and the persisted active selection remain available.
This operation is separate from removing a model from the cache.

### Request

```json
{
  "model": "qwen35-9b-q4k-m"
}
```

The optional `model` identifies the model the caller expects to be active.
If another client activated a different model, the server returns
`409 Conflict` and leaves that model loaded. An empty request object unloads
the current model. Repeating an unload while idle succeeds.

### Response

```json
{
  "object": "model.unload",
  "unloaded_from_gpu": true,
  "active": false,
  "deleted_model": false
}
```

Prompt-reuse state and active-context counters are cleared. Generation returns
`503` until a model is activated again. Vulkan/Metal unload waits for generation
to finish; CUDA/ROCm returns `409` while requests are active or queued so a
model change cannot interrupt their use of GPU resources.

---

## POST /v1/models/remove

Remove an installed managed model from the local cache. If the target model is
currently loaded, the request fails by default. Set `force: true` to unload it
first and then remove the cached files.

### Request

```json
{
  "model": "qwen35-9b-q4k-m",
  "force": false
}
```

### Response

```json
{
  "object": "model.remove",
  "id": "qwen35-9b-q4k-m",
  "removed": true,
  "unloaded_from_gpu": false,
  "cleared_active_selection": true,
  "deleted_model": true,
  "deleted_manifest": true,
  "removed_dir": true
}
```

Without `force`, a loaded target returns `409 Conflict` and the cached files
remain untouched. After a forced remove, the server may have no model loaded;
generation endpoints then return `503` until another model is activated or the
server is restarted with a model.

---

## GET /health

Server health check for monitoring and local UI state.

### Response

```json
{
  "status": "ok",
  "model": "Qwen 3.5 9B Q4_K_M",
  "supports_model_management": true,
  "supports_model_unload": true,
  "supports_model_pull": true,
  "active_requests": 1,
  "queued_requests": 0,
  "uptime_seconds": 3600,
  "gpu_memory_used_bytes": 10100000000,
  "gpu_memory_budget_bytes": 34359738368,
  "gpu_memory_weights_bytes": 8900000000,
  "gpu_memory_runtime_bytes": 1200000000,
  "gpu_context_reserved_bytes": 536870912,
  "gpu_context_active_bytes": 131072,
  "gpu_context_tokens": 128,
  "gpu_context_capacity_tokens": 4096
}
```

`model` is `"none"` when the server is running without a loaded model.

---

## Error Responses

Errors use an OpenAI-style envelope:

```json
{
  "error": {
    "message": "Field 'messages' is required",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

| Status | Type | When |
|--------|------|------|
| 400 | `invalid_request_error` | Malformed JSON, missing required fields, invalid parameters, unknown model id for management operations, unsupported or non-fitting model activation. |
| 400 | `context_length_exceeded` | Prompt exceeds the active model context capacity. |
| 404 | `model_not_found` | Generation requested an unknown or not-installed managed model. |
| 404 | `not_found` | Unknown endpoint. |
| 409 | `invalid_request_error` | GPU already reserved, another model download is in progress, or removing the loaded model without `force: true`. |
| 409 | `model_changed` or `model_busy` | Unload targeted a model that is no longer active, or CUDA/ROCm model management was requested with generation active or queued. |
| 501 | `unsupported_operation` | Model-management endpoint used on a build/backend without management support. |
| 503 | `service_unavailable` | Generation requested while no model is loaded. |
| 500 | `internal_error` | Tokenization, GPU, inference, or response-size failure. |

## CORS And Connection Handling

- All endpoints include `Access-Control-Allow-Origin: *`.
- `OPTIONS` requests return `200 {}` for CORS preflight.
- Streaming responses use HTTP/1.1 chunked transfer encoding (`Content-Type: text/event-stream`, `Cache-Control: no-cache`, `Connection: keep-alive`, `Transfer-Encoding: chunked`).
- Client disconnection during streaming stops generation promptly.
- Non-streaming JSON responses send `Connection: close`. The HTML routes (`/`, `/chat`) also send `Connection: close`.

## Server Configuration

```text
./zig-out/bin/zinc [options]
  -m, --model <path>       Path to GGUF model file
  --model-id <id>          Managed model id from the built-in catalog/cache
  -p, --port <port>        Server port (default: 8080; `zinc chat` defaults to 9090)
  -d, --device <id>        GPU device index (default: auto; prefers discrete Vulkan GPU)
  -c, --context <size>     Context length (default: 4096)
  --parallel <n>           Max queued/concurrent HTTP requests (default: 4)
  --kv-quant <bits>        TurboQuant KV cache compression: 0/2/3/4 (default: 0=off)
  --prompt <text>          Single prompt mode (CLI only; does not start the server)
  -n, --max-tokens <n>     CLI generation token cap
  --chat                   Apply the model chat template in CLI prompt mode
  --raw                    Do not auto-apply chat templates in CLI prompt mode
```

## Supported Models

ZINC loads GGUF models and supports the quantization families used by the
managed catalog:

| Format | Supported | Notes |
|--------|-----------|-------|
| Q4_K | Yes | Primary K-quant path |
| Q5_K | Yes | K-quant path |
| Q6_K | Yes | K-quant path |
| Q4_0 | Yes | Inkling 2 Mini on AMD RDNA4 |
| Q8_0 | Yes | Common attention and projection path |
| F16 | Yes | Small tensors and some model data |
| F32 | Yes | Baseline/scalar tensor path |

Run `./zig-out/bin/zinc model list` or `GET /v1/models` for the exact catalog
entries that are supported and fit on the current machine.
