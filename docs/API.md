# ZINC Serving API Reference

ZINC exposes a local OpenAI-compatible HTTP API. Point OpenAI-style clients at
the server by changing the base URL.

For a first connection, follow [Run models and connect apps](RUNNING_ZINC.md#start-the-api).
This page is the request/response reference. Examples use port 9090 and the
installed `zinc` command. Use the loaded model ID from `/v1/models` when copying
requests; older ROCm/CUDA builds use GGUF filenames, and 0.1.2 native serving
uses catalog IDs for known installed models.

Jump to [backend coverage](#backend-coverage), [endpoints](#endpoints),
[chat requests](#post-v1-chat-completions), [model discovery](#get-v1-models)
or [errors](#error-responses). Client recipes are in [OpenClaw](/zinc/docs/openclaw/)
and [OpenCode](/zinc/docs/opencode/).

## Backend coverage

Each executable contains one backend. Check `zinc --version` before selecting
an interface or copying server settings. HIP / ROCm is the preferred AMD RDNA
path; Vulkan serves Intel and provides an AMD alternative. Metal is a macOS
preview, and CUDA is experimental. The table describes Linux 0.1.2 and the
current Metal preview and experimental CUDA builds.

| Feature           | Vulkan / Metal                  | ROCm / CUDA                      |
| ----------------- | ------------------------------- | -------------------------------- |
| Chat responses    | JSON default; SSE optional      | SSE default; JSON optional       |
| Raw responses     | JSON default; SSE optional      | SSE default; JSON optional       |
| Chat sampling     | Supported                       | Greedy                           |
| Serving           | Serialized; requests queue      | Batched; Gemma MoE uses one slot |
| Default context   | Auto-sized                      | 2,048 per slot                   |
| Models            | Catalog and hot switching       | Installed-model switching        |
| Browser chat      | Included                        | HIP only                         |
| Prefix reuse      | Eligible Vulkan models          | Unavailable                      |
| Chat stop strings | Applied                         | Ignored                          |
| Health details    | Model, queue, uptime and memory | Status, backend and context      |

Both chat APIs accept `max_completion_tokens` and `max_tokens`; tools depend on
the model template. Vulkan/Metal default to 256 output tokens. ROCm/CUDA default
to the CLI `-n` value (256 unless overridden), and `-c 0` still means a
2,048-token context per slot. Native health returns 503 after GPU execution
failure. On ROCm/CUDA, a zero output
limit uses the remaining context budget; send a positive limit for bounded
portable requests.

ROCm setup, runtime dependencies and optional KV/MTP controls are documented in
the [ROCm guide](ROCM.md). Vulkan and ROCm use independent device lists, driver
stacks and benchmark results. Release qualification is narrower than source
compatibility; consult the [binary release guide](GETTING_STARTED.md).

See [HIP browser chat and model lifecycle](#hip-browser-chat-and-model-lifecycle)
for details. For portable clients, set `stream` explicitly, get the loaded model
from `/v1/models`, and check `/v1/capabilities` for optional behavior.

## Capability discovery

The Linux 0.1.1 releases include `GET /v1/capabilities` and the additional
health fields below. Current Metal/CUDA source builds implement this schema
too, while remaining preview/experimental backends. These fields are absent
in 0.1.0. Check `zinc --version`; on older servers, fall back to `/health` and
`/v1/models` when the endpoint returns 404.

```bash
curl --fail http://localhost:9090/v1/capabilities
```

All four backends return the same schema:

| Field            | Meaning                                                                                                                                                                                                           |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema_version` | `1`; clients must tolerate additive fields.                                                                                                                                                                       |
| `build`          | `version`, `commit`, `target`, `optimize`, and compiled `backend`.                                                                                                                                                |
| `backend`        | `vulkan`, `metal`, `rocm`, or `cuda`.                                                                                                                                                                             |
| `active_model`   | Loaded ID matching `/v1/models`, or null when idle. HIP 0.1.2 uses catalog IDs for known models. Older native builds use GGUF basenames.                                                                          |
| `context_length` | Live capacity per request/slot, or null when idle.                                                                                                                                                                |
| `capabilities`   | Boolean `chat_completions`, `text_completions`, `chat_streaming`, `text_streaming`, `model_management`, `model_switching`, `browser_ui`, `sampling`, and `chat_session_reuse`; `tool_calls` is `model_dependent`. |

Healthy `/health` responses include these fields and preserve their existing
backend-specific fields. Vulkan/Metal's legacy `model` remains a display name;
use `active_model` as the request ID. An unloaded server can be
healthy while `active_model` and `context_length` are null. Native execution
failure still returns HTTP 503. Discovery describes features; it does not
execute a request or qualify a model's output quality.

`chat_session_reuse` reflects the currently loaded eligible model and is false
when absent or unqualified. On older servers, use the existing `/health` and
`/v1/models` fields, set `stream` explicitly, and treat missing capability flags
as unknown. See [Run models and connect apps](RUNNING_ZINC.md) for the complete workflow.

## Status and compatibility limits

- Vulkan/Metal serialize generation behind one engine lock. `--parallel` bounds
  requests in flight; it does not multiply decode throughput. ROCm/CUDA use their
  batched worker; model and memory constraints still apply.
- `/v1/embeddings`, `/v1/audio/*`, image input, fine-tuning and the OpenAI batch
  API are not implemented. Batched GPU decode is separate from the batch API.
- The raw `/v1/completions` route ignores sampling controls. Use chat when you
  need sampling on Vulkan/Metal. ROCm/CUDA remain greedy on both routes.
- `/health` advertises `capabilities.chat_session_reuse` only on Vulkan with an
  eligible attention-only model. Clients may send a per-conversation
  `session_id` when this is true. Missing or false capability means full
  requests without assumed cache support; changed or displaced prefixes fall
  back to full prefill.
- Vulkan 0.1.1 uses canonical prefix replay by default for Intel
  Xe2 Gemma MoE. `ZINC_CHAT_PREFIX_ROLLBACK=0` disables it; `=1` opts other
  eligible attention-only models into this control. It preserves only the exact
  shared prefix and recomputes the changed suffix, including a dropped Gemma
  thinking header. It requires the same resident session/model and live KV
  pages, excludes recurrent and MTP state, and leaves at least one prompt token
  to refresh logits. Other models keep their existing transcript-reuse default.
- Tool calling depends on the model's chat template and output quality. Qwen
  uses ChatML/JSON tool envelopes; Gemma uses its native tool protocol. ZINC
  returns tool calls and accepts tool results; the client executes the tools.

## Endpoints

| Method | Path                   | Available on                            |
| ------ | ---------------------- | --------------------------------------- |
| POST   | `/v1/chat/completions` | All four backends                       |
| POST   | `/v1/completions`      | All four backends                       |
| GET    | `/v1/capabilities`     | Linux 0.1.1+; current Metal/CUDA source |
| GET    | `/v1/models`           | All four backends; different detail     |
| GET    | `/health`              | All four backends; different detail     |
| POST   | `/v1/models/pull`      | Vulkan / Metal                          |
| POST   | `/v1/models/activate`  | Vulkan, HIP, Metal, experimental CUDA   |
| POST   | `/v1/models/unload`    | Vulkan, HIP, Metal, experimental CUDA   |
| POST   | `/v1/models/remove`    | Vulkan / Metal                          |
| GET    | `/` or `/chat`         | Vulkan, HIP and Metal                   |
| GET    | `/stats`               | ROCm / CUDA                             |
| POST   | `/generate`            | ROCm / CUDA; raw completion alias       |

Unavailable endpoints return 404; native endpoint coverage is version-specific.
Set `stream` explicitly for both
completion routes. The coverage matrix above describes their response defaults.

## HIP browser chat and model lifecycle

HIP 0.1.2 serves browser chat at `GET /` and `GET /chat`. HIP 0.1.2 and current
experimental CUDA support switching among installed models from the startup
model's directory and managed cache. It never accepts an arbitrary path
from an HTTP client. Known files use catalog IDs; generation also accepts the
previous GGUF-basename alias. Unknown siblings remain unavailable for activation.

`/health` and `/v1/capabilities` preserve the versioned discovery schema and
advertise model switching/unload. `/v1/models` retains `created: 0` and `owned_by`
and lists installed candidates, with context/thinking/MTP details for the active
model only. When unloaded, `active_model` and `context_length` are null and the
legacy `model` alias is empty. Generation then returns 503 until a model is loaded.

A switch releases the old model before allocating its replacement. Invalid
metadata, incompatible cache settings and conservative context/slot memory bounds
are checked first. Load failures attempt to restore the previous model; a failed
restore leaves the server unloaded. Active/queued generation holds a lifecycle
lease, so conflicting native management requests return 409 instead of freeing
in-use resources. The built-in UI keeps a draft across unload/reload and offers
these operations only when the running server advertises them.

HIP sampling is greedy. Use the `zinc model` CLI to download or remove cached
weights on HIP and CUDA.

## POST /v1/chat/completions

Generate a chat completion from a message list.

### Request

```json
{
  "model": "qwen35-9b-q4k-m",
  "session_id": "chat-123",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is the capital of France?" }
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "enable_thinking": false,
  "stream": true
}
```

| Field                   | Type             | Default          | Description                                                                                                                                                                                                            |
| ----------------------- | ---------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`                 | string           | current model    | Activates an installed model if different, when the server advertises model switching.                                                                                                                                 |
| `session_id`            | string           | empty            | Optional prompt-reuse key only when `/health` advertises `capabilities.chat_session_reuse`; other backends ignore it.                                                                                                  |
| `messages`              | array            | required         | OpenAI-style messages. `content` may be a string, `null`, or an array of text content blocks. Non-text blocks are ignored.                                                                                             |
| `max_completion_tokens` | integer or null  | 256              | Maximum generated tokens before context-budget clamping. Supported by modern OpenAI-compatible clients, including OpenCode 2.                                                                                          |
| `max_tokens`            | integer or null  | 256              | Legacy alias of `max_completion_tokens`.                                                                                                                                                                               |
| `temperature`           | float            | 0.0              | Sampling temperature. `0` uses greedy decoding. Values are clamped to `0..2`.                                                                                                                                          |
| `top_p`                 | float            | 1.0              | Nucleus sampling threshold, clamped to `0..1`.                                                                                                                                                                         |
| `enable_thinking`       | boolean          | model default    | Controls reasoning for compatible templates. Vulkan/Metal also apply the managed catalog's stability policy. Tool requests normally suppress thinking unless explicitly enabled; model/template behavior still varies. |
| `stream`                | boolean          | backend-specific | Enable Server-Sent Events streaming. Set explicitly: Vulkan/Metal default to `false`, ROCm/CUDA to `true`.                                                                                                             |
| `tools`                 | array            | empty            | OpenAI function-tool definitions. Rendered using the model's supported Qwen/ChatML or native Gemma tool format.                                                                                                        |
| `tool_choice`           | string or object | `auto`           | `"none"` suppresses tool injection. `"auto"` allows tool selection. `"required"` can prefill a single tool where the template supports it. Function-object choices currently fall back to automatic choice.            |

Both chat token-limit names use the same budget. If both are non-null, their
values must agree; conflicting limits return HTTP 400 before inference. Omitting
both uses 256 on Vulkan/Metal or the server's CLI `-n` value on ROCm/CUDA
(also 256 unless overridden). Raw `/v1/completions` continues to use `max_tokens`.

ROCm/CUDA also retain legacy `n_predict` as a fallback when neither standard
chat limit is set, or when raw `max_tokens` is absent. Standard token-limit
fields take precedence over `n_predict`; it is not a third consistency alias.
Vulkan/Metal do not use this native compatibility field.

Supported message roles:

| Role        | Description                                                                                   |
| ----------- | --------------------------------------------------------------------------------------------- |
| `system`    | System prompt.                                                                                |
| `user`      | User message.                                                                                 |
| `assistant` | Previous assistant response. May include historical `tool_calls`.                             |
| `tool`      | Tool result message with `tool_call_id`; rendered in the model's native tool-response format. |

### Tool Calling

Tool calling is enabled by default and can be disabled with:

```bash
ZINC_TOOL_CALLING=0 zinc --model-id qwen35-9b-q4k-m
```

ZINC never executes tools. It only renders tool definitions into the prompt,
parses model-emitted Qwen or native Gemma tool envelopes, returns structured
`tool_calls`, and accepts subsequent `role: "tool"` result messages from the
client. The [JSON-prefill tool guide](JSON_PREFILL_TOOL_CALLS.md) documents
required-single-tool prefixes, their CPU regressions and verified API controls.

Example request:

```json
{
  "model": "qwen35-9b-q4k-m",
  "messages": [{ "role": "user", "content": "What time is it in UTC?" }],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_time",
        "description": "Return the current time for a timezone.",
        "parameters": {
          "type": "object",
          "properties": {
            "timezone": { "type": "string" }
          },
          "required": ["timezone"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "stream": false
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
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35-9b-q4k-m",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 64,
    "stream": false
  }'

curl -N http://localhost:9090/v1/chat/completions \
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

Per-request MTP control and completion metadata are available in HIP 0.1.2
and current experimental CUDA.

Native HIP/ROCm and experimental CUDA serving report `supports_mtp`,
`mtp_default_enabled`, `mtp_request_control`, and `mtp_unavailable_reason` on
`/health` and the active `/v1/models` entry. Support requires a compatible dense
NextN model, one request slot, and successfully prepared GPU state. ROCm defaults
on; CUDA defaults off. An explicit `ZINC_MTP=0` remains an operator kill switch.

Current ROCm source builds also expose this control for
[Inkling 2 Mini DSpark](RUNNING_ZINC.md#inkling-2-mini-on-amd-rdna4)
when the server loads its companion GGUF. Inkling drafting defaults off.

When `mtp_request_control` is true, chat and raw completions accept nullable
`enable_mtp`: `true` requests MTP and returns 400 if unavailable; `false` selects
ordinary decode for this request; omission or `null` follows the server default.
Each queued request keeps its own choice. Vulkan/Metal report request control as
false and reject explicit overrides. Vulkan retains its existing global MTP
policy and verifier; a missing support/default flag is unknown, not false.

Capability describes the initialized model/runtime, not a claim that a reply
used MTP. The chat UI uses `mtp_default_enabled` until the user chooses a mode;
unsupported or unreported capability never silently becomes an opt-in.

On native HIP/ROCm and CUDA, the final SSE chunk (before `[DONE]`) and
non-streaming completion contain an
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

The request/JSON example below applies to all backends. Set `stream` explicitly:
Vulkan/Metal default to JSON; ROCm/CUDA default to SSE. Linux Vulkan 0.1.1 and
current Metal source support raw SSE; the 0.1.0 Vulkan archives return JSON there.

Generate from a raw text prompt without applying a chat template.

### Request

```json
{
  "model": "qwen35-9b-q4k-m",
  "prompt": "The capital of France is",
  "max_tokens": 64,
  "stream": false
}
```

| Field                          | Type    | Default          | Description                                                                    |
| ------------------------------ | ------- | ---------------- | ------------------------------------------------------------------------------ |
| `model`                        | string  | current model    | May activate an installed model when the server advertises model switching.    |
| `prompt`                       | string  | required         | Raw text prompt.                                                               |
| `max_tokens`                   | integer | 256              | Maximum generated tokens before context-budget clamping.                       |
| `temperature`                  | float   | accepted         | Parsed for compatibility but not applied by this endpoint yet.                 |
| `stream`                       | boolean | backend-specific | Enable SSE text chunks. Vulkan/Metal default to `false`; ROCm/CUDA to `true`.  |
| `stream_options.include_usage` | boolean | `false`          | On Vulkan/Metal, append a separate usage event before `[DONE]` when streaming. |

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

With `stream: true`, raw chunks use `choices[].text` and object
`text_completion`. They do not add assistant-role deltas or apply chat templates.
The final choice has empty text and a `stop` or `length` finish reason. When
`stream_options.include_usage` is true, a following event has empty `choices`
and total token usage. The stream ends with `data: [DONE]`.

Vulkan/Metal stream the same greedy continuation as their JSON route, including
safe handling of Unicode characters split across tokens. EOG tokens count toward
usage but are omitted from text. An unfinished Unicode character at the output
budget is omitted in both formats. Vulkan checks for a disconnected reader
during prefill and between decode tokens, then releases the generation slot.
The Metal preview checks during decode; it does not interrupt active prefill.

---

## GET /v1/models

Stable 0.1.1 CUDA/ROCm builds return a minimal list containing the loaded model:

```json
{
  "object": "list",
  "data": [
    {
      "id": "model.gguf",
      "object": "model",
      "created": 0,
      "owned_by": "zinc",
      "context_length": 32768
    }
  ]
}
```

In 0.1.1 native builds, `id` is the file basename including its extension,
and `context_length` is the capacity of one request slot. HIP 0.1.2 lists the
installed candidates and exposes activation and unload as described under
[HIP browser chat and model lifecycle](#hip-browser-chat-and-model-lifecycle).

Vulkan/Metal builds provide the fuller response below.

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

| Status | Type                            | When                                                                                                                                                  |
| ------ | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 400    | `invalid_request_error`         | Malformed JSON, missing required fields, invalid parameters, unknown model id for management operations, unsupported or non-fitting model activation. |
| 400    | `context_length_exceeded`       | Prompt exceeds the active model context capacity.                                                                                                     |
| 404    | `model_not_found`               | Generation requested an unknown or not-installed managed model.                                                                                       |
| 404    | `not_found`                     | Unknown endpoint.                                                                                                                                     |
| 409    | `invalid_request_error`         | GPU already reserved, another model download is in progress, or removing the loaded model without `force: true`.                                      |
| 409    | `model_changed` or `model_busy` | Unload targeted a model that is no longer active, or CUDA/ROCm model management was requested with generation active or queued.                       |
| 501    | `unsupported_operation`         | Model-management endpoint used on a build/backend without management support.                                                                         |
| 503    | `service_unavailable`           | Generation requested while no model is loaded.                                                                                                        |
| 500    | `internal_error`                | Tokenization, GPU, inference, or response-size failure.                                                                                               |

## CORS And Connection Handling

- All endpoints include `Access-Control-Allow-Origin: *`.
- `OPTIONS` requests return `200 {}` for CORS preflight.
- Streaming responses use HTTP/1.1 chunked transfer encoding (`Content-Type: text/event-stream`, `Cache-Control: no-cache`, `Connection: keep-alive`, `Transfer-Encoding: chunked`).
- Client disconnection during streaming stops generation promptly.
- Non-streaming JSON responses send `Connection: close`. The HTML routes (`/`, `/chat`) also send `Connection: close`.

## Server Configuration

```text
zinc [options]
  -m, --model <path>       Path to GGUF model file
  --model-id <id>          Managed model id from the built-in catalog/cache
  -p, --port <port>        Server port (default: 8080; `zinc chat` defaults to 9090)
  -d, --device <id>        Vulkan/Metal index; ROCm/CUDA use native visibility environment variables
  -c, --context <size>     Vulkan/Metal: auto; ROCm/CUDA: 2048 tokens per slot
  --parallel <n>           Requests in flight / native request slots (default: 4; see coverage above)
  --kv-quant <bits>        Reserved compatibility option; no runtime effect
  --prompt <text>          Single prompt mode (CLI only; does not start the server)
  -n, --max-tokens <n>     CLI generation token cap
  --chat                   Apply the model chat template in CLI prompt mode
  --raw                    Do not auto-apply chat templates in CLI prompt mode
```

## Supported Models

ZINC loads GGUF models and supports the quantization families used by the
managed catalog:

| Format | Supported | Notes                                |
| ------ | --------- | ------------------------------------ |
| Q4_K   | Yes       | Primary K-quant path                 |
| Q5_K   | Yes       | K-quant path                         |
| Q6_K   | Yes       | K-quant path                         |
| Q8_0   | Yes       | Common attention and projection path |
| F16    | Yes       | Small tensors and some model data    |
| F32    | Yes       | Baseline/scalar tensor path          |

Run `zinc model list` for catalog entries and fit checks. Vulkan/Metal
also expose them through `GET /v1/models`; ROCm/CUDA return only the loaded model.
Use the backend's qualification report for the exact model, context and cache settings tested.
