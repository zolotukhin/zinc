# Running ZINC

> **Current status**: ZINC is still experimental and under active development. The CLI path is the most dependable way to run it today. Server mode exists, but you should still treat it as evolving.

This page is the shortest path from “the project builds” to “I can actually use it.”

## Build the binary

From the repo root:

```bash
zig build -Doptimize=ReleaseFast
```

The executable will be here:

```bash
./zig-out/bin/zinc
```

## Run a preflight check first

This is the fastest way to answer "does this machine, Vulkan stack, and model look runnable at all?"

```bash
# General machine + Vulkan + shader preflight
./zig-out/bin/zinc --check

# Recommended on RDNA4 shells
export RADV_PERFTEST=coop_matrix

# Check one exact GGUF file
./zig-out/bin/zinc --check -m /path/to/model.gguf

# Or check one managed model from the built-in catalog
./zig-out/bin/zinc --check --model-id qwen35-35b-a3b-q4k-xl
```

On the shared RDNA4 test node, the equivalent command is:

```bash
# SSH to the shared RDNA4 node
cd /root/zinc

# General preflight
./zig-out/bin/zinc --check

# Managed model compatibility by catalog id
./zig-out/bin/zinc --check --model-id qwen35-35b-a3b-q4k-xl

# Exact GGUF file check
./zig-out/bin/zinc --check -m /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

`--check` prints its progress as numbered sections and ends with a summary and verdict. The five sections are:

- `Host Environment`: OS and the current shell's `RADV_PERFTEST`
- `Linux AMD Prerequisites`: Mesa and GECC / RAS checks on Linux
- `Runtime Assets`: compiled shader directory and required `.spv` files
- `Vulkan Device`: loader init, GPU enumeration, selected device, and tuning summary
- `Model File`: GGUF metadata plus an estimated VRAM fit for the current engine

The model section is the most operationally useful one. It reports:

- `Tensor upload`: exact device-local weight bytes derived from the GGUF tensors
- `VRAM fit`: estimated device-local total against the selected GPU's VRAM
- `KV cache`: current estimate for the active runtime, which is capped to a `4096` token context in today's engine
- `GPU SSM state`: persistent device-local SSM state when the architecture uses it
- `host-visible staging`: mapped/readback buffers, reported separately from device-local VRAM

Important assumptions behind the fit estimate:

- it reflects the current single-GPU runtime, not multi-GPU sharding
- it reflects the current engine's `4096` KV cap even if the GGUF advertises a much larger context window
- it excludes Vulkan allocation alignment, descriptor pools, query pools, and driver overhead

Exit behavior:

- `READY [OK]` and `READY WITH WARNINGS [WARN]` exit with code `0`
- `NOT READY [FAIL]` exits non-zero

One common point of confusion: `RADV_PERFTEST` is checked in the shell where you run `--check`. A plain SSH shell can warn even if your long-running service sets `RADV_PERFTEST=coop_matrix` correctly.

### Sample outputs from the shared RDNA4 node

These outputs were captured on **March 29, 2026** from `/root/zinc` on the shared RDNA4 host.

General preflight:

```bash
# Command
./zig-out/bin/zinc --check

# Key output
Summary       : 7 ok, 1 warn, 0 fail, 1 skip
Verdict       : READY WITH WARNINGS [WARN]
```

Managed 35B catalog check:

```bash
# Command
./zig-out/bin/zinc --check --model-id qwen35-35b-a3b-q4k-xl

# Key output
Managed model: Qwen3.5 35B-A3B UD Q4_K_XL (qwen35-35b-a3b-q4k-xl) [OK]
VRAM fit (catalog): 21.41 / 31.86 GiB device-local (headroom 10.45 GiB) [OK]
Summary       : 9 ok, 1 warn, 0 fail, 0 skip
Verdict       : READY WITH WARNINGS [WARN]
```

Exact GGUF file check:

```bash
# Command
./zig-out/bin/zinc --check -m /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf

# Key output
Model: /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf [OK]
Tensor upload: 20.70 GiB device-local weights
VRAM fit: 21.41 / 31.86 GiB device-local (headroom 10.45 GiB) [OK]
Summary       : 9 ok, 1 warn, 0 fail, 0 skip
Verdict       : READY WITH WARNINGS [WARN]
```

The warning in all three cases came from the shell environment, not from model fit:

```bash
# The shared host shell did not have this set during the sample run
RADV_PERFTEST: Not set in current shell [WARN]
```

## Inspect the managed model catalog

The built-in managed catalog only lists models ZINC has explicitly revalidated for specific GPU profiles. The default `model list` view is strict: a model only appears there if it is both tested for the detected GPU profile and estimated to fit current VRAM.

```bash
# Show models that are tested for the detected GPU profile
# and estimated to fit current VRAM
./zig-out/bin/zinc model list

# Show the full built-in catalog even if local GPU probing is unavailable
./zig-out/bin/zinc model list --all

# Show the current managed default, if one is configured
./zig-out/bin/zinc model active
```

Example `./zig-out/bin/zinc model list` output on Apple Silicon:

```bash
Detected GPU profile: apple-silicon

ID                             Released     Status      Fit    Installed   Active   Notes
gpt-oss-20b-q4k-m              2025-06-25   supported   yes    yes         no       tested + exact fit
qwen3-8b-q4k-m                 2025-04-29   supported   yes    yes         yes      tested + exact fit
qwen36-35b-a3b-q4k-xl          2026-04-15   supported   yes    no          no       tested + exact fit
gemma4-31b-q4k-m               2026-04-02   supported   yes    no          no       tested + catalog fit
```

For machine-readable output (useful for AI agents and scripts):

```bash
./zig-out/bin/zinc model list --json
```

If you want ZINC to manage downloads and the default startup model for you:

```bash
# Download one managed model into the local cache
./zig-out/bin/zinc model pull qwen3-8b-q4k-m

# Mark it as the default managed model for future runs
./zig-out/bin/zinc model use qwen3-8b-q4k-m

# Inspect the current managed default
./zig-out/bin/zinc model active

# Remove a cached managed model
./zig-out/bin/zinc model rm qwen3-8b-q4k-m

# Force-unload it from the local server first if it is still active there
./zig-out/bin/zinc model rm --force qwen3-8b-q4k-m
```

`model rm` is conservative by default: if the local ZINC server still has that model loaded in GPU memory, the command refuses and leaves the cache untouched. Use `--force` to have the local server unload it first. If your server uses a non-default port, add `--port <port>` before `model rm`.

## Run a single prompt from the terminal

This is the best first command to try:

```bash
export RADV_PERFTEST=coop_matrix
./zig-out/bin/zinc \
  -m /path/to/model.gguf \
  --prompt "The capital of France is"
```

That runs a single CLI inference pass and prints the build, model, GPU, prefill, and decode logs directly to the terminal.

## Pick a specific GPU

If your machine exposes multiple Vulkan devices, choose one explicitly:

```bash
./zig-out/bin/zinc \
  -m /path/to/model.gguf \
  -d 0 \
  --prompt "Hello"
```

Useful when the Vulkan device list includes integrated graphics, llvmpipe, or multiple AMD devices.

## Change context length

```bash
./zig-out/bin/zinc \
  -m /path/to/model.gguf \
  -c 8192 \
  --prompt "Summarize this document"
```

Use a larger context only if the model and GPU VRAM budget can support it.

## Enable TurboQuant KV compression

ZINC exposes KV cache quantization through `--kv-quant`.

```bash
./zig-out/bin/zinc \
  -m /path/to/model.gguf \
  --kv-quant 3 \
  --prompt "Explain page attention in one paragraph"
```

Supported values:

- `0` = disabled
- `2` = 2-bit
- `3` = 3-bit
- `4` = 4-bit

If you are experimenting with constrained VRAM, `3` is the natural first value to try.

## Run ZINC as a server

When you omit `--prompt`, ZINC starts an HTTP server with a built-in chat interface and an OpenAI-compatible API:

```bash
./zig-out/bin/zinc \
  -m /path/to/model.gguf \
  -p 8080
```

Once you see `Server listening on 0.0.0.0:8080`, open your browser:

- **Chat UI**: [http://localhost:8080/](http://localhost:8080/) — a ChatGPT-like interface with streaming, markdown rendering, syntax highlighting, and copy buttons on code blocks.
- **Health**: [http://localhost:8080/health](http://localhost:8080/health)

### RDNA4 test-node deploy

If you want to rebuild and restart the shared RDNA4 test server from your current checkout, use the helper script:

```bash
./scripts/deploy_rdna4_server.sh
```

That script uses `.env` for `ZINC_HOST`, `ZINC_USER`, and `ZINC_PORT`, syncs the repo to `/root/zinc`, rebuilds it, restarts port `9090`, and finishes with a remote `/health` check.

Useful flags:

- `--no-sync` when the remote checkout is already current
- `--no-build` when you only want a restart
- `--no-restart` when you only want to push code and compile
- `--no-healthcheck` when you will validate separately

Current limitation: generation is still serialized behind one engine lock. Overlapping clients complete cleanly, `/health` stays responsive, and queued generation work is reported, but decode itself is not yet parallel.

## OpenAI-compatible API

The server exposes `/v1` endpoints that work as a drop-in replacement for OpenAI and llama-server clients.

### Streaming chat completion

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "enable_thinking": true,
    "stream": true,
    "max_tokens": 256
  }'
```

### Non-streaming chat completion

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "enable_thinking": true,
    "max_tokens": 32
  }'
```

`enable_thinking` only has an effect when the active model's chat template exposes a thinking toggle. In the built-in chat UI, the Thinking checkbox is shown only for models that report this capability.

### List models

```bash
# Show managed models, fit status, install state, and the active entry
curl http://localhost:8080/v1/models
```

### Activate a managed model in a running server

```bash
# Switch the running server to an installed managed model
curl http://localhost:8080/v1/models/activate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-q4k-m"
  }'
```

### Remove a managed model from a running server

```bash
# Remove an installed model if it is not currently loaded
curl http://localhost:8080/v1/models/remove \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-q4k-m"
  }'

# Force the server to unload it first if it is still active
curl http://localhost:8080/v1/models/remove \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-q4k-m",
    "force": true
  }'
```

Without `force`, a loaded target returns `409` and ZINC leaves the cached files untouched.

### Use with OpenAI SDKs

Node.js:

```javascript
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://localhost:8080/v1", apiKey: "unused" });
const stream = await client.chat.completions.create({
  model: "qwen3.5-35b",
  messages: [{ role: "user", content: "Hello!" }],
  stream: true,
});
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || "");
}
```

### All endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Built-in chat interface |
| GET | `/health` | Server status, loaded model, active/queued requests, uptime |
| GET | `/v1/models` | List managed models, fit status, install state, and the active entry |
| POST | `/v1/models/activate` | Activate an installed managed model |
| POST | `/v1/models/remove` | Remove a cached managed model, optionally unloading it first |
| POST | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |
| POST | `/v1/completions` | Text completion |

The full API contract is documented in [Serving HTTP API](/zinc/docs/api).

## Export the decode graph

ZINC can emit decode-graph artifacts directly from GGUF metadata. The JSON report is model-aware: it includes estimated bytes moved, FLOPs, hotspot ranking, and bottleneck hints for each node.

```bash
./zig-out/bin/zinc \
  -m /path/to/model.gguf \
  --graph-report decode-graph.json \
  --graph-dot decode-graph.dot
```

That is useful for:

- graph inspection
- hotspot and bandwidth analysis
- custom tooling
- Graphviz rendering
- Bun-based HTML dashboards

The fastest way to review the report is the HTML renderer:

```bash
bun run graph:render -- decode-graph.json decode-graph-report.html
open decode-graph-report.html   # macOS
# xdg-open decode-graph-report.html   # Linux
```

That dashboard is built by [`tools/render_graph_report.ts`](../tools/render_graph_report.ts). It groups the raw graph into:

- top hotspots
- bottleneck mix
- layer pressure
- op mix
- critical-path excerpt
- searchable full node table

The renderer only needs the JSON report. The DOT file is optional and is still useful when you want the full dependency graph structure.

If you have Graphviz installed, you can still render the DOT file:

```bash
dot -Tsvg decode-graph.dot -o decode-graph.svg
```

## What a healthy run looks like

A normal run usually includes lines like:

```bash
info(vulkan): Selected GPU 0: AMD Radeon Graphics
info(loader): Loading model: /path/to/model.gguf
info(forward): Prefill complete: ...
info(forward): Generated ... tok/s
```

That is the simplest operational checklist:

1. Vulkan device selected
2. Model loaded
3. Prefill finished
4. Decode happened

## Common early mistakes

### Vulkan is present, but the wrong device is selected

Use `-d` to pick the GPU you actually want.

### The model path is wrong

ZINC expects a GGUF file, not a Hugging Face directory or a framework checkpoint.

### The machine builds, but runtime is unstable

That usually points to environment issues before it points to the model:

- wrong Vulkan driver stack
- unsupported hardware path
- missing `RADV_PERFTEST=coop_matrix` on RDNA4
- model too large for the available VRAM

## Fast troubleshooting commands

```bash
./zig-out/bin/zinc --help
./zig-out/bin/zinc --check -m /path/to/model.gguf
zig build test --summary all
vulkaninfo --summary
```

Those four commands answer most “why is this not working?” questions:

- is the CLI wired correctly?
- does the machine/model preflight look sane?
- does the codebase still pass its test suite?
- does Vulkan actually see the GPU?

## Related docs

- [Getting started](/zinc/docs/getting-started)
- [Hardware requirements](/zinc/docs/hardware-requirements)
- [Serving HTTP API](/zinc/docs/api)
- [RDNA4 tuning](/zinc/docs/rdna4-tuning)
