# Run models and connect apps

After [installing ZINC and running a first prompt](GETTING_STARTED.md), choose
[browser chat](#browser-chat), a [terminal prompt](#terminal-prompts), or the
[HTTP API](#start-the-api). These commands work with stable Linux 0.1.1,
including [structured discovery](#structured-discovery).

## Browser chat

Vulkan and Metal include a browser interface:

```bash
zinc chat --model-id qwen35-9b-q4k-m -c 4096 --port 9090
```

Keep that terminal running. Open [localhost:9090](http://localhost:9090/) if the
browser does not open automatically. Press Ctrl-C in the server terminal to
stop it. ROCm/CUDA in stable 0.1.1 provide the API only; use an external chat client instead.
Stop this server before starting another ZINC command on the same GPU or port.

### Development: HIP browser chat

Current HIP source builds serve the same browser interface for the loaded
startup model. After [building ROCm](ROCM.md#build-from-source), use
`zinc chat --model-id qwen38-27b-q4k-m -c 65536 --parallel 1` on a card with
enough memory. It supports streaming and the model's thinking toggle;
model switching and sampling remain unavailable. This is not in the published
0.1.1 archive. See [context and KV choices](ROCM.md#context-kv-cache-and-mtp)
before increasing the window. Over SSH, use `--no-browser` and forward port 9090.

## Terminal prompts

Run one prompt and exit:

```bash
zinc --model-id qwen35-9b-q4k-m -c 4096 --chat \
  --prompt "Explain KV caching in three sentences." -n 128
```

`--chat` applies the model's conversation template. Add `--system-prompt
"Answer directly."` when you need a system instruction. Use `--raw` for a
plain text continuation instead. `-n` limits generated tokens; `-c` sets the
context capacity, including the prompt and output. See `zinc --help` for the
installed version's options.

## Start the API

This is the common starting point for apps and agent clients. With the model
already downloaded, start one server and leave it running:

```bash
zinc serve --model-id qwen35-9b-q4k-m -c 16384 --parallel 1 --port 9090
```

The 16K context leaves room for tools and conversation history. Check its
[memory requirements](HARDWARE_REQUIREMENTS.md#model-memory) first; increase it
only when the model and GPU can support it. A model's advertised native window
does not reserve that much memory on your machine.

In another terminal, check readiness and discover the loaded model:

```bash
curl --fail http://localhost:9090/health
curl --fail http://localhost:9090/v1/models
```

Expect HTTP 200 and a loaded model with a positive `context_length`.
Vulkan/Metal list catalog entries: select the row with `active: true`.
ROCm/CUDA list only the loaded model, with its GGUF filename as the `id`.
A healthy but idle Vulkan/Metal server is not ready to generate until a model
is loaded.

## Connect an app

| Client setting      | Value                                                                  |
| ------------------- | ---------------------------------------------------------------------- |
| Provider / protocol | OpenAI-compatible **Chat Completions**                                 |
| API base URL        | `http://localhost:9090/v1`                                             |
| Model               | The loaded `id` from `/v1/models`                                      |
| Context             | The loaded model's `context_length`, not its advertised maximum        |
| API key             | `unused` if the client requires a nonempty value for this local server |

The server listens on all interfaces and has no built-in API authentication.
Keep its port private; for remote access, use an [SSH tunnel](OPENCODE.md#use-your-own-endpoint)
or an authenticated gateway. A placeholder API key does not protect the server.

To try a request directly, replace `LOADED_MODEL_ID` with that exact discovery
ID. The other fields work across backends:

```bash
curl --fail-with-body http://localhost:9090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "LOADED_MODEL_ID",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "enable_thinking": false,
    "max_tokens": 64,
    "stream": false
  }'
```

Expect a JSON response with `choices[0].message.content`. For streaming, set
`"stream": true` and add `-N` to curl. Always set `stream`: Vulkan/Metal default
to JSON, while ROCm/CUDA default to SSE. The [HTTP API reference](API.md) covers
request fields, tool calls, error responses and backend differences. An
OpenAI-compatible client must support Chat Completions; compatibility with
other OpenAI endpoints is not implied.

## Connect an agent

ZINC generates responses; the client runs file and shell tools on the client
machine. Start with a small task and verify the files and project tests yourself.
The 0.6B installation-test model is not suitable for these agent workflows.

| Client   | Follow this guide                   | What its check proves                                          |
| -------- | ----------------------------------- | -------------------------------------------------------------- |
| OpenClaw | [Setup, check and run](OPENCLAW.md) | Client configuration, server health and context; no generation |
| OpenCode | [Setup, check and run](OPENCODE.md) | Live text, tool-call and streaming protocol requests           |

Both guides use standalone helpers and an existing project directory; neither
requires a ZINC checkout. OpenClaw discovers the model and context; OpenCode uses an explicit model ID
and can discover its context. Repeat
setup when changing the server model or context, and check the client guide's
recorded model results before choosing a larger configuration. A successful
protocol check alone does not prove that a task was completed correctly.

## Models and cache

Use managed IDs for routine downloads and runs, including on Metal:

```bash
zinc model list
zinc model list --all
zinc model list --json
zinc model pull qwen35-9b-q4k-m
zinc model use qwen35-9b-q4k-m
zinc model active
```

`use` sets the default for future launches. It does not switch an already
running native server. Vulkan/Metal support live model management through the
[API](API.md#post-v1-models-activate); restart ROCm/CUDA to change models.

For a GGUF you already have, replace `--model-id` with `-m`:

```bash
zinc --check -m /path/to/model.gguf -c 4096
zinc -m /path/to/model.gguf -c 4096 --chat --prompt "Hello" -n 64
```

A GGUF outside the catalog still needs a supported architecture, quantization
and enough memory. A framework checkpoint or model directory is not a GGUF file.

To remove cached weights after stopping the process that uses them:

```bash
zinc model rm qwen35-9b-q4k-m
```

Vulkan/Metal can refuse removal while a local server has the model loaded.
`--force` requests that server unload it first; for a custom server port, use
`zinc --port 9090 model rm --force <id>`. Native servers have no unload endpoint;
stop them before removing their model.

## Device, context and KV settings

| Setting                              | Use                                                                                                                                                                     |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-d <index>`                         | Override Vulkan GPU selection; inspect `vulkaninfo --summary` and `zinc --check`.                                                                                       |
| `ROCR_VISIBLE_DEVICES`               | Select visible ROCm devices before launch; native HIP indices differ from Vulkan.                                                                                       |
| `CUDA_VISIBLE_DEVICES`               | Select visible CUDA devices before launch.                                                                                                                              |
| `-c <tokens>` / `--context <tokens>` | Explicit context capacity. Vulkan/Metal otherwise auto-size; ROCm/CUDA default to 2,048 per slot, including with `-c 0`.                                                |
| `--parallel 1`                       | One request slot. Vulkan/Metal serialize generation; a higher value does not multiply decode throughput. Native backends can batch, subject to model and memory limits. |
| `--port <port>`                      | Explicit port. Plain server mode defaults to 8080; `chat` defaults to 9090.                                                                                             |

ROCm/CUDA ignore `-d`; confirm the selected native device with `zinc --check`.
`--kv-quant` is a reserved compatibility option with **no runtime effect**.
For opt-in ROCm Q8 KV storage, use the [native-context guide](ROCM_NATIVE_CONTEXT.md);
keep default precision for release qualification comparisons.

## Troubleshooting

| Symptom                               | Next check                                                                                                                                         |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `zinc: command not found`             | Add `~/.local/bin` to PATH; rerun `zinc --version`.                                                                                                |
| No GPU / runtime initialization fails | Run `zinc --check`; use the relevant [driver check](HARDWARE_REQUIREMENTS.md#driver-checks).                                                       |
| Missing Vulkan shaders                | Keep the installed `bin/` and `share/` directories together, or rerun the installer.                                                               |
| Wrong GPU                             | Check the backend and use the device controls above. Vulkan and native device indices are independent.                                             |
| Out of memory                         | Stop your other model processes, reduce `-c`, use one slot or choose a smaller model. `--check` is an estimate, not proof of successful inference. |
| Address in use / GPU reserved         | Stop the earlier server you started before launching its replacement.                                                                              |
| API returns SSE instead of JSON       | Set `"stream": false` explicitly.                                                                                                                  |
| Client cannot find a model            | Copy the loaded ID from `/v1/models`; native backends use a GGUF filename. Rerun client setup after model changes.                                 |
| Agent exceeds context                 | Match the live context, shorten instructions/tool output, and use the client helper's bounded settings.                                            |
| Slow, missing or corrupt output       | Capture version, command and server logs. On AMD RDNA4 Vulkan, confirm `RADV_PERFTEST=coop_matrix`; then check model/backend qualification.        |

If it still fails, [file an issue](https://github.com/zolotukhin/zinc/issues) with
`zinc --version`, `zinc --check`, GPU/driver, model ID, context and a reproducing
prompt. Remove local credentials and private machine details from shared logs.

## Structured discovery

Available in ZINC 0.1.1 and later. Check the installed `zinc --version` and
`zinc --help` before applying these commands to an older installation:

```bash
zinc --version --json
zinc --check --json --model-id qwen35-9b-q4k-m -c 16384 --parallel 1
zinc serve --model-id qwen35-9b-q4k-m -c 16384 --parallel 1 --port 9090
```

`serve` never opens a browser. `zinc chat --no-browser` keeps the Vulkan/Metal
browser UI available without opening it automatically. Port defaults remain
8080 for `serve` and 9090 for `chat`.

JSON diagnostics write one document to stdout and runtime logs to stderr.
`result` is `ready`, `ready_with_warnings` or `not_ready`; exit codes are 0
without failures and 1 on failure. Native memory estimates include context and
serving slots, but lazy runtime allocations can exceed the stated reserve.
Use the same `-c` value you intend to serve. For an installed model, Vulkan
estimates its declared window when `-c` is omitted. Follow the metadata/GPU
preflight with a real request to validate allocation and inference.

```bash
curl --fail http://localhost:9090/v1/capabilities
```

Schema version 1 reports build identity, backend, `active_model`,
`context_length` and capability flags, also included in healthy `/health`
responses. Idle Vulkan/Metal servers report null model/context. Tool support is
model-dependent. See [capability fields](API.md#capability-discovery).

On older 0.1.0 servers this endpoint returns 404: use `/health` and `/v1/models` instead.
Treat absent capability flags as unknown. In particular, only use chat session
reuse when the loaded server explicitly advertises it.

## Working on ZINC itself

Use [Development](DEVELOPMENT.md) for building, testing, graph exports and
remote deployment, plus the repository
[AGENTS.md](https://github.com/zolotukhin/zinc/blob/main/AGENTS.md) and
[codebase map](CODEBASE_MAP.md) for source changes. Those tools are unnecessary
for installing a release or connecting an app.
