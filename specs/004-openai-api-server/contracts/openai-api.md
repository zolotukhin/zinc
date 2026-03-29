# API Contract: OpenAI-Compatible Endpoints

## POST /v1/chat/completions
Request: {model, messages[], stream?, max_tokens?, temperature?, top_p?, stop?}
Response (stream): SSE events with ChatCompletionChunk objects, ending data: [DONE]
Response (non-stream): Single ChatCompletion with usage

## POST /v1/completions
Request: {model, prompt, stream?, max_tokens?, temperature?, top_p?, stop?}
Response: Same pattern as chat but with text_completion objects

## GET /v1/models
Response: {object:"list", data:[{id, object:"model", created, owned_by:"zinc"}]}

## GET /health
Response 200: {status:"ok", model, active_requests, max_parallel, uptime_seconds}
Response 503: {status:"loading"}

## Errors
{error: {message, type, code}} — 400/404/429/500/503
