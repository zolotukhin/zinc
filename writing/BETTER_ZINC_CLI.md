# Better ZINC CLI

Status: draft

This document outlines a managed-model workflow for ZINC that behaves more like a Docker image runtime than a one-off binary that needs a raw `.gguf` path every time.

The goal is to let a user:

- detect the current GPU automatically
- see only models that ZINC actually supports and has tested on that GPU class
- download those models into a local cache directory
- verify file integrity with a published SHA-256 before activation
- run a chosen cached model from CLI or server mode
- switch the active model from the CLI or the built-in chat UI

This is an operating model and implementation plan, not a code patch.

## Why

Current ZINC behavior is still path-centric:

- users pass `-m /path/to/model.gguf`
- the server loads one model at startup
- `/v1/models` reports the currently loaded model, not a managed catalog
- the chat UI reflects that one loaded model and cannot switch it

That is workable for development, but it is not a good product surface. A user should not need to know where a GGUF file lives, which model variant fits their GPU, or whether a file is valid before they can use ZINC.

## Product Shape

ZINC should behave like this:

1. detect the current GPU and memory budget
2. resolve a tested compatibility profile
3. show a catalog of supported model images for that profile
4. allow pull into a local cache
5. verify checksum before marking the model installed
6. activate one cached model at a time
7. let CLI and chat use that active model by default
8. allow switching to another installed compatible model without manually editing paths

For now, ZINC still loads one active model into memory at a time. This is the right MVP because the runtime is still effectively single-engine.

## Non-Goals For V1

- arbitrary user-provided download URLs
- multi-GPU sharding
- loading multiple active models in memory at once
- background continuous hot-swapping while requests are mid-generation
- peer-to-peer or registry auth flows

Those can come later. V1 should be opinionated and controlled.

## User Flows

### Flow 1: Discover what this machine can run

Example:

```bash
zinc model list
```

Expected behavior:

- ZINC detects the local GPU using the existing Vulkan and GPU-detection path
- ZINC computes the current compatibility profile, for example `amd-rdna4-32gb`
- ZINC shows only models that are marked `supported` and `tested` for that profile
- ZINC annotates whether each model is already installed, whether it fits current VRAM, and whether it is currently active

Example output:

```text
Detected GPU profile: amd-rdna4-32gb

ID                             Status      Fit    Installed   Active   Notes
qwen35-35b-a3b-q4k-xl          supported   yes    yes         yes      tested on RDNA4 32 GB
qwen35-14b-q4k                 supported   yes    no          no       tested on RDNA4 16 GB+
llama33-8b-q4k                 supported   yes    yes         no       tested on RDNA4 12 GB+
```

### Flow 2: Pull a supported model into local cache

Example:

```bash
zinc model pull qwen35-14b-q4k
```

Expected behavior:

- ZINC resolves the catalog entry
- ZINC downloads into a temporary file in the model cache
- ZINC reports progress and source URL clearly
- ZINC computes SHA-256
- ZINC compares SHA-256 with the published digest in the catalog
- only after a match does ZINC rename into the final cache path and mark the model installed

Example output:

```text
Resolving model: qwen35-14b-q4k
Downloading: https://...
Received: 8.42 GiB
Verifying sha256...
sha256 verified
Installed: ~/.cache/zinc/models/qwen35-14b-q4k/model.gguf
```

### Flow 3: Activate a cached model

Example:

```bash
zinc model use qwen35-14b-q4k
```

Expected behavior:

- ZINC confirms the model is installed
- ZINC re-runs fit validation against the current GPU profile
- ZINC writes the active-model pointer to local config
- if server mode is running, ZINC can either:
  - switch immediately through an admin API, or
  - instruct the user to switch through the server endpoint

### Flow 4: Run from the active cached model

Example:

```bash
zinc --prompt "Hello"
zinc serve
```

Expected behavior:

- if `--model-id` is not provided, CLI and server use the active model from config
- if there is no active model, ZINC tells the user how to select one
- existing direct path mode remains available for advanced users

### Flow 5: Switch model inside chat UI

Expected behavior:

- the chat UI shows an installed-model selector
- switching models calls a local ZINC admin endpoint
- the switch drains or blocks generation during the model transition
- after switch, the UI starts a fresh conversation because tokenizer and system behavior may differ

## Core Concepts

### 1. Supported Model Catalog

ZINC needs a first-party catalog of known-good models.

Each entry should include:

- `id`
- `display_name`
- `family`
- `format`
- `quantization`
- `size_bytes`
- `sha256`
- `download_url`
- `license`
- `homepage`
- `tested_profiles`
- `minimum_vram_bytes`
- `default_context_length`
- `recommended_for_chat`
- `status`

`status` should be one of:

- `supported`
- `experimental`
- `hidden`
- `deprecated`

For the default UX, only `supported` models that match the current GPU profile should be shown.

### 2. GPU Compatibility Profile

ZINC already detects:

- vendor family
- GPU generation
- VRAM
- bandwidth
- Vulkan features

That should be normalized into a stable profile key such as:

- `amd-rdna4-32gb`
- `amd-rdna4-16gb`
- `amd-rdna3-16gb`

Catalog support should be expressed against those stable profile keys, not against ad hoc runtime strings.

### 3. Local Model Cache

ZINC needs a managed cache directory for pulled models.

Suggested resolution:

- Linux:
  - `$XDG_CACHE_HOME/zinc/models`
  - fallback: `~/.cache/zinc/models`
- macOS:
  - `~/Library/Caches/zinc/models`

Suggested layout:

```text
<cache-root>/
  catalog.json
  downloads/
    <model-id>.partial
  models/
    <model-id>/
      model.gguf
      manifest.json
```

`manifest.json` should record:

- model id
- installed timestamp
- file size
- sha256
- source URL
- catalog version used for installation

### 4. Active Model Pointer

ZINC should store the selected model separately from the model cache.

Suggested config location:

- Linux:
  - `$XDG_CONFIG_HOME/zinc/active-model.json`
  - fallback: `~/.config/zinc/active-model.json`
- macOS:
  - `~/Library/Application Support/zinc/active-model.json`

Contents:

```json
{
  "active_model_id": "qwen35-35b-a3b-q4k-xl",
  "selected_at": "2026-03-29T23:00:00Z"
}
```

The active pointer should refer to a model id, not a path.

## CLI Surface

### New commands

```bash
zinc model list
zinc model list --all
zinc model pull <model-id>
zinc model use <model-id>
zinc model installed
zinc model remove <model-id>
zinc model active
```

### Runtime selection

Keep existing raw-path mode:

```bash
zinc -m /path/to/model.gguf --prompt "Hello"
```

Add managed-model selection:

```bash
zinc --model-id qwen35-14b-q4k --prompt "Hello"
zinc serve --model-id qwen35-14b-q4k
```

Selection order should be:

1. explicit `--model-id`
2. explicit `-m/--model` raw path
3. active model config
4. fail with a clear setup message

That keeps backward compatibility while moving users toward the managed path.

## Server And Chat Behavior

### `/v1/models`

This endpoint should stop pretending there is only one model in the world.

It should return:

- supported models for this machine
- whether each one is installed
- whether it is active
- whether it currently fits the detected GPU budget

That makes it usable for the chat UI and for local admin tooling.

### New local admin endpoints

Suggested endpoints:

- `POST /v1/models/pull`
- `POST /v1/models/activate`
- `GET /v1/models/active`

These are ZINC-specific operational endpoints, not strict OpenAI API compatibility endpoints.

### Chat UI

The built-in chat page should:

- fetch the model catalog from `/v1/models`
- show only installed compatible models in a selector
- visually mark the active model
- allow switching the active model
- clear the current conversation after a successful switch
- disable send while switch is in progress

The selector should sit in the existing top bar, near the current model badge.

## Switching Semantics

Because the current server runtime is still one-engine-at-a-time, model switching must be explicit and serialized.

V1 behavior should be:

- acquire the server generation lock
- reject or queue new requests during switch
- wait for the active request to finish, or refuse switching if configured `busy`
- unload current model resources
- load tokenizer, model weights, and inference engine for the new model
- mark new model active
- release the lock

Operationally, this is more important than making switching instant.

### Chat-side consequence

A conversation should not silently continue across a model switch. The UI should reset the local thread after activation and show:

```text
Switched to qwen35-14b-q4k. Started a new conversation.
```

## Integrity And Safety

### Checksum verification

Every catalog model must ship with a published SHA-256.

Pull flow:

1. download to temp file
2. fsync temp file if practical on the target platform
3. compute SHA-256 over the full file
4. compare with catalog digest
5. fail closed on mismatch
6. only then rename into final cache path

On mismatch:

- remove the partial file
- print expected and actual SHA prefixes
- do not register the model as installed

### Fit validation

Before activation and before server startup, ZINC should reuse the current fit-estimation logic from `--check` and validate:

- weight upload size
- KV cache footprint
- GPU SSM state
- runtime buffer estimate
- headroom threshold

A pulled model may be cached but still not activatable on the current machine.

## Catalog Source Of Truth

V1 should use a checked-in catalog, not a remote dynamic registry.

That keeps the system deterministic and testable.

Suggested source:

- `assets/models/catalog.json`

That file should be updated only when:

- the model download URL is confirmed
- SHA-256 is confirmed
- the model has been tested on the profile(s) it claims

This is the key rule: ZINC only advertises models that the project has actually tested.

## Proposed Data Shapes

### Catalog entry

```json
{
  "id": "qwen35-35b-a3b-q4k-xl",
  "display_name": "Qwen 3.5 35B A3B UD Q4_K_XL",
  "download_url": "https://...",
  "sha256": "abc123...",
  "size_bytes": 22234567890,
  "minimum_vram_bytes": 34359738368,
  "tested_profiles": ["amd-rdna4-32gb"],
  "status": "supported",
  "default_context_length": 4096,
  "recommended_for_chat": true
}
```

### Runtime view

```json
{
  "id": "qwen35-35b-a3b-q4k-xl",
  "display_name": "Qwen 3.5 35B A3B UD Q4_K_XL",
  "supported_on_current_gpu": true,
  "fits_current_gpu": true,
  "installed": true,
  "active": true,
  "status": "supported"
}
```

## Implementation Plan

### Phase 1: Catalog and cache plumbing

- add a model catalog module
- add cache-dir and config-dir resolution helpers
- add manifest read and write helpers
- add SHA-256 verification helpers
- add `zinc model list`
- add `zinc model active`

### Phase 2: Pull and activate

- add `zinc model pull <id>`
- add `zinc model use <id>`
- add fit validation before activation
- add active-model config persistence
- add `--model-id` runtime selection

### Phase 3: Server and chat integration

- replace single-entry `/v1/models` response with catalog-aware response
- add `POST /v1/models/activate`
- wire chat model selector
- clear local conversation after switch
- ensure switching is serialized under the existing generation lock

### Phase 4: Operational hardening

- add resumable downloads if needed
- add stale partial cleanup
- add richer progress output
- add better health reporting around model load and active model

## Test Plan

### Unit tests

- cache directory resolution
- config directory resolution
- catalog parsing
- checksum verification
- manifest read and write
- active model config read and write
- GPU profile filtering
- fit rejection before activation

### Integration tests

- list shows only supported models for the mocked GPU profile
- pull succeeds on matching SHA-256
- pull fails and cleans up on mismatch
- activating an uninstalled model fails cleanly
- server startup uses active model when no explicit override is passed
- chat switch endpoint changes active model and resets conversation state

### Smoke tests

- fresh machine: list, pull, activate, prompt
- installed machine: switch from model A to model B
- incompatible machine: cached model exists but activation is rejected

## Open Questions

- Do we want `zinc model pull` to support mirrors from day one, or only one canonical URL per model?
- Should `zinc serve` permit automatic switch requests while busy, or return `409 busy` and require retry?
- Should the chat UI show only installed models, or also show supported-but-not-installed models with a download action?

## Recommended MVP Cut

If we keep the first release tight, the MVP should be:

- checked-in catalog
- GPU-aware `zinc model list`
- local cache directory
- SHA-256 verified `zinc model pull`
- active model persistence
- `--model-id` support
- `/v1/models` returns installed and active state
- chat dropdown for installed compatible models
- serialized active-model switching

That gives ZINC the “managed runtime” feel without pretending it already has multi-model serving or a remote registry.
