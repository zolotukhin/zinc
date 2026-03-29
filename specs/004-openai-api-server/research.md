# Research: OpenAI-Compatible API Server

## R1: HTTP Parsing — Manual HTTP/1.1 on std.net (existing scaffold)
## R2: SSE Streaming — Manual chunked transfer encoding
## R3: JSON — std.json for parsing, manual string building for responses
## R4: Chat Template — Read from GGUF metadata, minimal ChatML parser
## R5: Threading — Single-threaded event loop with non-blocking I/O
## R6: KV Cache — Fixed-size page pool, per-request page table
## R7: Batching — Round-robin per-sequence decode in existing loop

See full research in the plan discussion. All decisions resolved.
