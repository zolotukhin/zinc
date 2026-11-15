# Connect an agent or application to ZINC

Use ZINC as a local model server. The client runs tools; ZINC generates model
responses. OpenClaw can connect as an agent platform, and OpenCode provides a
coding client. Neither client needs the ZINC source tree.

## Stable Linux 0.1.0 workflow

Install the backend for your GPU using [Binary Installation](BINARY_RELEASES.md).
The stable release contains one backend per executable. GPU drivers and models
are separate from the installation. Metal is a preview; CUDA is experimental.

Inspect the installed command, check the GPU, and select a model:

```bash
zinc --version
zinc --check
zinc model list --json
zinc model pull qwen35-9b-q4k-m
```

`model list --json` reports qualification and memory fit separately. A model
fitting in memory does not establish tool quality. Qwen 3.5 9B is a starting
point for the recorded client tasks; the 0.6B installation-test model is not
recommended for agent tasks. Check the [hardware guide](HARDWARE_REQUIREMENTS.md)
and model-specific [OpenClaw](OPENCLAW.md) or [OpenCode](OPENCODE.md) results.

Start the server in one terminal; keep it running while using the client:

```bash
zinc --model-id qwen35-9b-q4k-m -c 16384 --parallel 1 --port 9090
```

Use the actual server context. A model's native context window does not reserve
that memory on your GPU. Check readiness and discover the loaded model:

```bash
curl --fail http://localhost:9090/health
curl --fail http://localhost:9090/v1/models
```

The API base URL is `http://localhost:9090/v1`. Set `stream` explicitly. Match the
loaded discovery ID, including ROCm's GGUF filename IDs. Do not assume every
backend can switch models or use sampling; see the [API coverage matrix](API.md#backend-coverage).

Download a standalone [OpenClaw helper](OPENCLAW.md#configure-a-project) or
[OpenCode helper](OPENCODE.md#configure-opencode), run setup, then run its check
command before submitting a task. The helpers discover the live context and
verify compatibility. Check task outcomes independently. Keep endpoint addresses
and credentials in runtime environment variables as described in those guides.

## Development: structured discovery and headless serving

The following additions require a development build newer than stable 0.1.0.
Check `zinc --help` before using them with a downloaded release. Existing server
commands above remain supported. See [Development](DEVELOPMENT.md) to build
current source; a source build identifies itself as `dev` unless explicitly
versioned by its packager.

```bash
zinc --version --json
zinc --check --json --model-id qwen35-9b-q4k-m -c 16384 --parallel 1
zinc serve --model-id qwen35-9b-q4k-m -c 16384 --parallel 1 --port 9090
```

`serve` never opens a browser. `zinc chat --no-browser` keeps browser chat
available while suppressing the automatic opener on Vulkan/Metal. Legacy server
mode and `serve` default to port 8080; `chat` defaults to 9090. Pass a port when
sharing configuration with another process.

```bash
curl --fail http://localhost:9090/v1/capabilities
```

Discovery returns schema version 1, build identity, backend, `active_model`,
`context_length`, and capability flags. The same fields are included in healthy
`/health` responses. An idle Vulkan/Metal server returns null for its active
model and context. Tool support is model-dependent; a capability flag is not
evidence that a model will complete your task.

For older servers returning 404, use `/health` and `/v1/models` and the documented
backend matrix. Treat absent capabilities as unknown; never infer support from
the backend's name alone. The helpers' live checks remain necessary.

JSON diagnostics emit one document on stdout, with status messages on stderr.
Exit code 0 means the preflight completed without failures; exit code 1 means a
diagnostic failed. `result` distinguishes `ready`, `ready_with_warnings` and
`not_ready`. Memory estimates include context and native serving slots but do
not prove that lazy runtime allocations or inference will succeed.

## Working on ZINC itself

For source changes, read the repository's
[AGENTS.md](https://github.com/zolotukhin/zinc/blob/main/AGENTS.md),
[Development](DEVELOPMENT.md), and [codebase map](CODEBASE_MAP.md). The generated
[Zig API](https://zolotukhin.ai/zinc/docs/zig-api/) describes internals; it is not
needed to install ZINC or connect a client.
