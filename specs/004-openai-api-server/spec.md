# Feature Specification: OpenAI-Compatible API Server

**Feature Branch**: `004-openai-api-server`
**Created**: 2026-03-28
**Status**: Draft
**Input**: User description: "HTTP server with OpenAI-compatible API (/v1/chat/completions, /v1/completions, /v1/models, /health) supporting SSE streaming, continuous batching, paged KV cache management, and concurrent requests. Drop-in replacement for llama-server with the same endpoint contracts. Must handle 4+ concurrent streaming chat completions at full decode throughput with no cross-contamination between requests."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Single Streaming Chat Completion (Priority: P1)

A developer sends a chat completion request to ZINC's API endpoint and receives a streaming response, token by token, in the same format as OpenAI's API. They can use their existing OpenAI SDK client code without modification — just change the base URL.

**Why this priority**: This is the fundamental value proposition. If a single request doesn't work correctly with standard tooling, nothing else matters. This story proves ZINC is a drop-in replacement.

**Independent Test**: Send `curl -N -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"qwen","messages":[{"role":"user","content":"What is 2+2?"}],"stream":true}'` and verify: (1) SSE events arrive token-by-token, (2) each event is valid JSON matching OpenAI's schema, (3) the final event is `data: [DONE]`, (4) the response is coherent.

**Acceptance Scenarios**:

1. **Given** ZINC is running with a loaded model, **When** a client sends a POST to `/v1/chat/completions` with `stream: true`, **Then** the server responds with `Content-Type: text/event-stream` and delivers tokens as `data: {...}\n\n` SSE events, ending with `data: [DONE]\n\n`.
2. **Given** a streaming request is in progress, **When** each SSE event is parsed, **Then** it contains a valid `ChatCompletionChunk` object with `id`, `object: "chat.completion.chunk"`, `created`, `model`, and `choices[0].delta.content`.
3. **Given** a client sends a non-streaming request (`stream: false` or omitted), **When** generation completes, **Then** the server responds with a single JSON `ChatCompletion` object containing the full generated text, token counts in `usage`, and a `finish_reason`.

---

### User Story 2 - Concurrent Requests Without Cross-Contamination (Priority: P1)

Multiple clients send chat completion requests simultaneously. Each client receives only their own response — no tokens from other requests leak into any response. The server handles all requests at full throughput without queuing or serializing them.

**Why this priority**: A server that can only handle one request at a time is barely more useful than the CLI. Concurrency with isolation is essential for any real deployment, from a local development proxy to a shared team server.

**Independent Test**: Launch 4 concurrent streaming requests with different prompts (e.g., "Count to 10 in French", "List 5 colors", "What is gravity?", "Name 3 planets"). Verify each response contains only content relevant to its prompt, no text from other requests appears in any stream, and all 4 complete successfully.

**Acceptance Scenarios**:

1. **Given** 4 concurrent streaming chat completion requests with distinct prompts, **When** all requests complete, **Then** each response contains only text relevant to its own prompt with zero cross-contamination.
2. **Given** 4 concurrent requests, **When** aggregate throughput is measured, **Then** it is at least 80% of 4x single-request throughput (minimal contention overhead).
3. **Given** one request encounters an error (e.g., malformed JSON), **When** it fails, **Then** the other concurrent requests continue unaffected.

---

### User Story 3 - Drop-In Replacement for llama-server (Priority: P2)

A developer currently using llama-server (llama.cpp's HTTP server) switches to ZINC by changing only the `--host` and `--port` flags. Their existing client code, scripts, and OpenAI SDK integrations continue working without modification.

**Why this priority**: Compatibility with the llama.cpp ecosystem means instant adoption by thousands of developers already using local LLM servers. No migration effort, no code changes.

**Independent Test**: Run the OpenAI Python SDK test suite against ZINC's server: `openai.ChatCompletion.create(model="qwen", messages=[...], base_url="http://localhost:8080/v1")`. Verify it returns a valid response. Repeat with streaming. Repeat with the `/v1/models` endpoint.

**Acceptance Scenarios**:

1. **Given** a client using the OpenAI Python SDK (`openai` package), **When** configured with `base_url="http://localhost:8080/v1"`, **Then** `chat.completions.create()` returns a valid response for both streaming and non-streaming modes.
2. **Given** a client using the OpenAI Node.js SDK (`openai` package), **When** configured with the ZINC server URL, **Then** streaming chat completions work identically to llama-server.
3. **Given** a tool that queries `/v1/models`, **When** it receives the response, **Then** it gets a valid `List[Model]` response containing the loaded model's name.

---

### User Story 4 - Health Monitoring and Operational Visibility (Priority: P3)

An operator deploying ZINC as a service can check server health, monitor active requests, and detect issues before they affect users. Standard monitoring tools (curl, Prometheus scrapers, load balancers) can probe the health endpoint.

**Why this priority**: Any production deployment needs health checking. Load balancers need to know if the server is ready. Operators need to know if the server is overloaded.

**Independent Test**: `curl http://localhost:8080/health` returns a JSON object with status, model name, active request count, and uptime.

**Acceptance Scenarios**:

1. **Given** the server is running and model is loaded, **When** a GET request is sent to `/health`, **Then** the response is `200 OK` with a JSON body containing at minimum `{"status": "ok"}`.
2. **Given** the server is starting but the model hasn't finished loading, **When** `/health` is probed, **Then** it returns `503 Service Unavailable` with `{"status": "loading"}`.
3. **Given** the server has active requests, **When** `/health` is queried, **Then** the response includes the count of active/pending requests.

---

### Edge Cases

- What happens when a client disconnects mid-stream? The server must detect the broken connection, stop generating tokens for that request, and free its KV cache pages.
- What happens when all KV cache pages are exhausted (too many concurrent long-context requests)? The server must reject new requests with `503 Service Unavailable` and a clear error message, not crash.
- What happens when a request specifies `max_tokens` exceeding the model's context length? The server must clamp to the available context and proceed, or return a 400 error.
- How does the server handle malformed JSON in the request body? Return `400 Bad Request` with a descriptive error message.
- What happens when a request specifies a model name that doesn't match the loaded model? Return `404 Not Found` with the available model listed.
- How does the server handle requests arriving faster than it can process them? Queue up to a configurable limit, then reject with `429 Too Many Requests`.
- What happens when `stop` sequences are specified? The server must check generated text against all stop sequences and halt generation when one is matched.
- How does the server handle the `temperature=0` case? It must produce deterministic output (greedy decoding) identical to the CLI mode.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Server MUST expose `POST /v1/chat/completions` accepting the OpenAI ChatCompletion request schema (model, messages, stream, max_tokens, temperature, top_p, stop, n, presence_penalty, frequency_penalty).
- **FR-002**: Server MUST expose `POST /v1/completions` accepting the OpenAI Completion request schema (model, prompt, stream, max_tokens, temperature, top_p, stop).
- **FR-003**: Server MUST expose `GET /v1/models` returning a list of available models in OpenAI's List[Model] format.
- **FR-004**: Server MUST expose `GET /health` returning server status, loaded model, and active request count.
- **FR-005**: Server MUST support SSE streaming for both chat completions and completions endpoints, using `Content-Type: text/event-stream` with `data: {json}\n\n` framing and a final `data: [DONE]\n\n` event.
- **FR-006**: Server MUST support non-streaming mode, returning the complete response as a single JSON object with a `usage` field (prompt_tokens, completion_tokens, total_tokens).
- **FR-007**: Server MUST handle at least 4 concurrent requests simultaneously, with independent decode state and no cross-contamination of generated text.
- **FR-008**: Server MUST detect client disconnection during streaming and immediately stop token generation for that request, freeing its resources.
- **FR-009**: Server MUST apply chat templates to convert the `messages` array into a single prompt string using the model's expected format (e.g., ChatML, Qwen chat template from GGUF metadata).
- **FR-010**: Server MUST support generation parameters: `max_tokens` (token limit), `temperature` (sampling temperature, 0 = greedy), `stop` (stop sequences), `top_p` (nucleus sampling).
- **FR-011**: Server MUST return proper HTTP error codes: 400 for malformed requests, 404 for unknown model, 429 for overload, 503 for not ready.
- **FR-012**: Server MUST include `id` (unique per request), `created` (unix timestamp), `model`, and `object` fields in all response objects, matching OpenAI's schema.
- **FR-013**: Server MUST log each request with: timestamp, client IP, endpoint, model, token counts, latency, and status code.
- **FR-014**: Server MUST manage KV cache pages across concurrent requests — allocating pages for new requests, freeing pages on completion or disconnection, and rejecting requests when pages are exhausted.

### Key Entities

- **Request**: An incoming HTTP request with a unique ID, prompt (or messages), generation parameters, and lifecycle state (pending, prefilling, generating, completed, cancelled). Each request owns a set of KV cache pages.
- **Session**: The server-side state for an active generation: request ID, token buffer, KV cache page allocation, current position, sampling state. Exists from request acceptance to completion/cancellation.
- **KV Cache Page**: A fixed-size block of GPU memory holding key and value vectors for a contiguous range of token positions. Pages are allocated per-request and freed on completion. The page pool is shared across all concurrent sessions.
- **Chat Template**: A formatting rule that converts a `messages` array (system, user, assistant turns) into a single prompt string that the model expects. Derived from model metadata.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A single streaming chat completion request delivers the first token within 500ms of the request being received (time to first token).
- **SC-002**: The server handles 4 concurrent streaming chat completions with aggregate decode throughput at least 80% of 4x single-request throughput.
- **SC-003**: Zero cross-contamination across 1000 concurrent request pairs (verified by automated test checking each response contains only tokens from its own prompt context).
- **SC-004**: The OpenAI Python SDK (`openai>=1.0`) successfully completes a chat completion (streaming and non-streaming) against ZINC without code changes beyond `base_url`.
- **SC-005**: Client disconnection during streaming frees all associated resources (KV cache pages, session state) within 1 second.
- **SC-006**: The server processes 100 sequential chat completion requests without memory leaks (RSS growth < 5% from request 1 to request 100).
- **SC-007**: All error responses conform to OpenAI's error schema: `{"error": {"message": "...", "type": "...", "code": "..."}}`.
- **SC-008**: Server startup to ready (health endpoint returns 200) completes within 30 seconds for the reference model.

## Assumptions

- The server runs on the same machine as the GPU and serves local or LAN clients. Public internet deployment (TLS, authentication, rate limiting per user) is out of scope for v1.
- Only one model is loaded at a time. Multi-model serving is out of scope.
- The chat template is derived from the GGUF model's metadata (`tokenizer.chat_template` field). If the model doesn't include a chat template, a reasonable default (ChatML) is used.
- Token counting in the `usage` field uses the model's native tokenizer (BPE from GGUF), not tiktoken or an external tokenizer.
- The `n` parameter (multiple completions per request) defaults to 1. Supporting `n > 1` is out of scope for v1.
- Function calling / tool use is out of scope for v1.
- Image/multimodal inputs are out of scope for v1.
- The continuous batching scheduler is a new component that coordinates between the HTTP server and the inference engine. It decides when to prefill vs decode, which requests to batch together, and manages KV cache page allocation.
- The existing single-request inference engine (forward pass) will be extended to support batched execution — processing multiple sequences in a single command buffer submission.
