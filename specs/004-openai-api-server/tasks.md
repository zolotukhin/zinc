# Tasks: OpenAI-Compatible API Server

**Input**: Design documents from `/specs/004-openai-api-server/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: US1 = Single Streaming Chat, US2 = Concurrent Requests, US3 = llama-server Drop-in, US4 = Health Monitoring

---

## Phase 1: Setup

**Purpose**: Server mode entry point and foundational HTTP infrastructure.

- [X] T001 Add server mode to main — when no --prompt flag, start HTTP server on configured port. Import and call Server.init + accept loop. File: src/main.zig
- [X] T002 Implement full HTTP/1.1 request parsing — read headers until \r\n\r\n, extract method, path, Content-Length, read body. Replace current stub readRequest. File: src/server/http.zig
- [X] T003 [P] Add HTTP response helpers — sendJson (status + JSON body), sendError (OpenAI error format), sendSseStart (streaming headers). File: src/server/http.zig

**Checkpoint**: Server starts, accepts connections, parses requests, sends JSON responses.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: SSE streaming, route dispatch, chat template, KV cache pages — needed by all stories.

- [X] T004 Implement SSE stream writer — chunked transfer encoding, write `data: {json}\n\n` events, `data: [DONE]\n\n`, detect client disconnect via write failure. File: src/server/sse.zig
- [X] T005 [P] Implement route dispatcher — match method+path to handler functions. Routes: POST /v1/chat/completions, POST /v1/completions, GET /v1/models, GET /health. Return 404 for unknown routes. File: src/server/routes.zig
- [X] T006 [P] Implement chat template application — read `tokenizer.chat_template` from GGUF metadata, apply ChatML-style template to messages array (insert role tags, special tokens). Fallback to default ChatML if template missing. File: src/model/tokenizer.zig
- [X] T007 Implement paged KV cache manager — page pool with fixed-size pages (256 tokens), alloc/free per request, free list, exhaustion detection (return error when pool empty). File: src/scheduler/kv_cache.zig
- [X] T008 Extend Scheduler with server integration — add methods: submitFromHttp (tokenize + allocate KV pages + create request), getSession (by slot_id), iterActiveSessions. Wire GenerationParams from parsed JSON. File: src/scheduler/scheduler.zig

**Checkpoint**: SSE writer works, routes dispatch, chat templates applied, KV pages managed.

---

## Phase 3: User Story 1 — Single Streaming Chat Completion (Priority: P1) 🎯 MVP

**Goal**: One client sends a chat completion request and receives streaming tokens via SSE.

**Independent Test**: `curl -N http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"qwen","messages":[{"role":"user","content":"Hello"}],"stream":true}'`

- [X] T009 [US1] Implement POST /v1/chat/completions handler — parse JSON request body (model, messages, stream, max_tokens, temperature, stop), validate required fields, return 400 on errors. Apply chat template to messages, tokenize prompt. File: src/server/routes.zig
- [X] T010 [US1] Implement streaming response path — for stream:true, create SSE writer, submit request to scheduler, run decode loop sending ChatCompletionChunk events per token, send [DONE] on completion. Include id, object, created, model, choices[0].delta fields. File: src/server/routes.zig
- [X] T011 [US1] Implement non-streaming response path — for stream:false or omitted, run full generation, build ChatCompletion response with complete message, usage (prompt_tokens, completion_tokens, total_tokens), finish_reason. File: src/server/routes.zig
- [X] T012 [US1] Implement single-request decode loop for server — adapt existing generate() to work with scheduler: call decodeStep per token, check shouldStop, write token to SSE stream or accumulate for non-streaming. Release KV pages on completion. File: src/compute/forward.zig
- [X] T013 [US1] Generate unique request IDs — format "chatcmpl-{hex}" using timestamp + counter. Include in all response objects. File: src/server/routes.zig

**Checkpoint**: Single streaming chat completion works end-to-end with curl.

---

## Phase 4: User Story 2 — Concurrent Requests (Priority: P1)

**Goal**: 4+ simultaneous streaming requests with no cross-contamination.

**Independent Test**: Launch 4 concurrent curl streams with different prompts, verify each gets only its own content.

- [ ] T014 [US2] Implement non-blocking connection accept loop — poll listener socket + all active connections. Accept new connections without blocking active streams. File: src/server/http.zig
- [ ] T015 [US2] Implement round-robin multi-sequence decode — iterate all active sessions per decode cycle: for each session, run one decodeStep, write token to its SSE stream. Interleave prefill and decode across requests. File: src/compute/forward.zig
- [ ] T016 [US2] Implement client disconnect detection — when SSE write fails (broken pipe), transition request to cancelled, free KV pages, remove session. Do not affect other sessions. File: src/server/sse.zig
- [ ] T017 [US2] Implement request queuing and 429 rejection — when all scheduler slots are full, queue up to 8 pending requests. If queue also full, return 429 Too Many Requests. Process queue when slots free up. File: src/scheduler/scheduler.zig
- [ ] T018 [US2] Add per-request KV cache isolation — each request gets its own page allocation from the shared pool. Verify no KV cache pages are shared between concurrent requests. Free all pages on request completion/cancellation. File: src/scheduler/kv_cache.zig

**Checkpoint**: 4 concurrent streaming requests complete correctly with no cross-contamination.

---

## Phase 5: User Story 3 — llama-server Drop-in (Priority: P2)

**Goal**: OpenAI SDK and llama-server clients work without code changes.

**Independent Test**: `python -c "from openai import OpenAI; c=OpenAI(base_url='http://localhost:8080/v1',api_key='x'); print(c.chat.completions.create(model='qwen',messages=[{'role':'user','content':'Hi'}]).choices[0].message.content)"`

- [X] T019 [US3] Implement POST /v1/completions handler — parse prompt (string), apply same generation logic as chat completions but without chat template. Return text_completion objects. File: src/server/routes.zig
- [X] T020 [US3] Implement GET /v1/models handler — return list containing the loaded model with id, object, created, owned_by fields. Model name derived from GGUF filename. File: src/server/routes.zig
- [X] T021 [US3] Add OpenAI SDK compatibility fields — ensure all response objects include required fields the SDK validates: id (string), object (exact type string), created (unix timestamp), model (string). Verify choices array format matches SDK expectations. File: src/server/routes.zig
- [X] T022 [US3] Handle optional parameters gracefully — ignore unknown request fields (forward compatibility). Apply defaults for missing optional fields (temperature=1.0, top_p=1.0, max_tokens=256). Return 400 only for truly invalid values, not missing optional fields. File: src/server/routes.zig
- [X] T023 [US3] Implement stop sequence detection — check generated text against stop[] array after each token. When matched, set finish_reason="stop" and halt generation. Handle multi-token stop sequences. File: src/compute/forward.zig

**Checkpoint**: OpenAI Python SDK chat.completions.create works for both streaming and non-streaming.

---

## Phase 6: User Story 4 — Health Monitoring (Priority: P3)

**Goal**: Operators can check server health and active request count.

**Independent Test**: `curl http://localhost:8080/health` returns JSON with status, model, active count.

- [X] T024 [US4] Implement GET /health handler — return status ("ok" or "loading"), model name, active_requests count from scheduler, max_parallel, uptime_seconds. Return 503 during model loading. File: src/server/routes.zig
- [X] T025 [US4] Add request logging — log each completed request with: timestamp, client IP, endpoint, model, prompt_tokens, completion_tokens, latency_ms, status_code. Use Zig's std.log. File: src/server/routes.zig

**Checkpoint**: Health endpoint works, request logs appear in stderr.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [X] T026 [P] Add CORS headers to all responses — Access-Control-Allow-Origin: *, required for browser-based clients. File: src/server/http.zig
- [ ] T027 [P] Implement graceful shutdown — catch SIGINT/SIGTERM, stop accepting new requests, drain active streams, clean up KV cache. File: src/main.zig
- [X] T028 Validate with OpenAI Python SDK integration test — script that tests streaming, non-streaming, models list, health, concurrent requests, error cases. File: tests/test_openai_sdk.py
- [ ] T029 Run quickstart.md validation — execute all test commands from quickstart.md and verify expected behavior. File: specs/004-openai-api-server/quickstart.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies
- **Phase 2 (Foundational)**: Depends on Phase 1 (HTTP infrastructure)
- **Phase 3 (US1 — MVP)**: Depends on Phase 2 (SSE, routes, chat template, KV cache)
- **Phase 4 (US2)**: Depends on Phase 3 (single request must work first)
- **Phase 5 (US3)**: Depends on Phase 3 (chat completions must work)
- **Phase 6 (US4)**: Depends on Phase 2 (routes only)
- **Phase 7 (Polish)**: Depends on all desired stories

### Parallel Opportunities

- T002, T003 can run in parallel (different functions in http.zig)
- T004, T005, T006 can run in parallel (different files)
- T009-T013 are sequential within US1 (handler → streaming → non-streaming → decode → IDs)
- T019, T020 can run in parallel (different endpoints)
- T026, T027 can run in parallel (different files)
- Phase 6 (US4) can start after Phase 2, in parallel with Phases 3-5

---

## Implementation Strategy

### MVP First (US1 Only)

1. Phase 1: HTTP server starts and accepts connections
2. Phase 2: SSE, routes, chat template, KV cache
3. Phase 3: Single streaming chat completion works
4. **STOP and VALIDATE**: curl test from quickstart.md
5. Demo-ready with one concurrent request

### Incremental Delivery

1. Phases 1+2 → Server infrastructure ready
2. Phase 3 → MVP: one chat completion with streaming
3. Phase 4 → Production: 4 concurrent requests
4. Phase 5 → Ecosystem: OpenAI SDK + /v1/completions
5. Phase 6 → Operations: health monitoring
6. Phase 7 → Production-ready: CORS, shutdown, testing
