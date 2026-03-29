# Running ZINC

> **Current status**: ZINC is still experimental and under active development. The CLI path is the most dependable way to run it today. Server mode exists, but you should still treat it as evolving.

This page is the shortest path from “the project builds” to “I can actually use it.”

## Build the binary

From the repo root:

```bash
zig build
```

The executable will be here:

```bash
./zig-out/bin/zinc
```

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

## OpenAI-compatible API

The server exposes `/v1` endpoints that work as a drop-in replacement for OpenAI and llama-server clients.

### Streaming chat completion

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "Hello!"}],
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
    "max_tokens": 32
  }'
```

### List models

```bash
curl http://localhost:8080/v1/models
```

### Use with OpenAI SDKs

Python:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

# Streaming
for chunk in client.chat.completions.create(
    model="qwen3.5-35b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

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
| GET | `/health` | Server status and loaded model |
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |
| POST | `/v1/completions` | Text completion |

The full API contract is documented in [Serving HTTP API](/zinc/docs/api).

## Export the decode graph

ZINC can emit structural graph artifacts directly from GGUF metadata.

```bash
./zig-out/bin/zinc \
  -m /path/to/model.gguf \
  --graph-report decode-graph.json \
  --graph-dot decode-graph.dot
```

That is useful for:

- graph inspection
- performance analysis
- custom tooling
- Graphviz rendering

If you have Graphviz installed:

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

ZINC expects a GGUF file, not a Hugging Face directory or PyTorch checkpoint.

### The machine builds, but runtime is unstable

That usually points to environment issues before it points to the model:

- wrong Vulkan driver stack
- unsupported hardware path
- missing `RADV_PERFTEST=coop_matrix` on RDNA4
- model too large for the available VRAM

## Fast troubleshooting commands

```bash
./zig-out/bin/zinc --help
zig build test
vulkaninfo --summary
```

Those three commands answer most “why is this not working?” questions:

- is the CLI wired correctly?
- does the codebase still pass its test suite?
- does Vulkan actually see the GPU?

## Related docs

- [Getting started](/zinc/docs/getting-started)
- [Hardware requirements](/zinc/docs/hardware-requirements)
- [Serving HTTP API](/zinc/docs/api)
- [RDNA4 tuning](/zinc/docs/rdna4-tuning)
