# Implementation Plan: OpenAI-Compatible API Server

**Branch**: `004-openai-api-server` | **Date**: 2026-03-28 | **Spec**: [spec.md](spec.md)

## Summary

Add an HTTP server to ZINC exposing OpenAI-compatible endpoints with SSE streaming, continuous batching, and concurrent request handling. Builds on existing scaffolding in src/server/ and src/scheduler/.

## Technical Context

**Language/Version**: Zig 0.14-dev
**Primary Dependencies**: Zig std.net, std.json, existing ZINC inference engine
**Testing**: `zig build test` + curl + OpenAI Python SDK
**Target Platform**: Linux, AMD RDNA4
**Project Type**: Inference server with HTTP API
**Performance Goals**: TTFT <500ms, 4 concurrent streams at 80%+ throughput
**Constraints**: No TLS, no auth, no multi-model

## Constitution Check

All 6 principles pass. This feature directly implements Principle V (Production Serving).

## Project Structure

```text
src/server/http.zig      # EXISTING: TCP listener (extend with full HTTP parsing)
src/server/sse.zig       # NEW: SSE stream writer
src/server/routes.zig    # NEW: route dispatcher, endpoint handlers
src/scheduler/scheduler.zig  # EXISTING: slot management (extend)
src/scheduler/request.zig    # EXISTING: request state machine
src/scheduler/kv_cache.zig   # NEW: paged KV cache manager
src/compute/forward.zig      # MODIFY: batched multi-sequence decode
src/model/tokenizer.zig      # MODIFY: chat template application
src/main.zig                 # MODIFY: server mode
```
