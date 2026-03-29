# Data Model: OpenAI API Server

## Entities

- **Request**: id, state (pending→prefilling→decoding→completed/cancelled), messages, prompt_tokens, generated_tokens, params, kv_pages, connection
- **Message**: role (system/user/assistant), content
- **GenerationParams**: max_tokens, temperature, top_p, stop, stream
- **KvPage**: page_id, owner_request_id, layer offsets, token_start, token_count
- **KvPagePool**: pages[], free_list[], page_size, total_pages
- **Session**: request, decode_state, sse_writer, last_activity

## Relationships

Server →1:N→ Session →1:1→ Request →owns→ KvPage[] ←pool← KvPagePool
