# ZINC binary distribution

Run GGUF language models on your GPU from the command line or through an
OpenAI-compatible local API.

This archive contains one platform/backend build. `VERSION.json` records its
version, source commit, backend, CPU/OS requirements, runtime dependencies,
and signing status. Use the matching release's notes and `validation.json`
for the GPU, driver, model, and context combinations actually tested.

Stable releases target Linux Vulkan/ROCm. Metal archives are unsigned
previews and are not included in the Linux release's hardware qualification.

## Run the extracted package

Install the GPU driver and backend runtime first. Keep `bin/` and `share/`
together; the executable finds its matching shaders relative to its location.
Zig, Bun, Python, and a source checkout are not needed to run this package.

From the extracted directory:

```bash
./bin/zinc --version
./bin/zinc --help
./bin/zinc --check
./bin/zinc --check --json
```

For AMD Vulkan, set `export RADV_PERFTEST=coop_matrix` before the GPU check
or inference. ROCm additionally needs compatible HIP, hipRTC, hipBLAS, and
the HIP/rocWMMA headers used for runtime compilation.

Models are downloaded separately. Choose a GGUF and context limit qualified
for your backend in the release notes. Replace the path in this example:

```bash
./bin/zinc -m /path/to/model.gguf -c 2048 --chat --prompt "Explain a ring buffer."
```

For an application or agent, start an explicit server without opening a browser:

```bash
./bin/zinc serve --model-id qwen35-9b-q4k-m -c 4096 --parallel 1 --port 9090
```

Pull that managed model first with `./bin/zinc model pull qwen35-9b-q4k-m`.
The server's `/v1/models` response advertises its current context and output
limits. Use those limits when configuring a client. OpenCode and OpenClaw setup
instructions are in the [agent and app guide](https://zolotukhin.ai/zinc/docs/agent-setup/).

Metal and Vulkan also provide browser chat:

```bash
./bin/zinc chat -m /path/to/model.gguf -c 2048
```

ROCm provides CLI/HTTP chat without the built-in browser UI or hot model
switching. See the [API coverage matrix](https://zolotukhin.ai/zinc/docs/api/#backend-coverage)
and backend guide before selecting optional KV, MTP, or concurrent serving
settings. Vulkan device indices and `RADV_PERFTEST` do not configure ROCm;
use its native visibility setting if you need to restrict the GPU.

## Install, upgrade, or get help

- [Binary installation and rollback](https://zolotukhin.ai/zinc/docs/getting-started/)
- [Hardware and runtime requirements](https://zolotukhin.ai/zinc/docs/hardware-requirements/)
- [CLI and server usage](https://zolotukhin.ai/zinc/docs/running-zinc/)
- [OpenAI-compatible API](https://zolotukhin.ai/zinc/docs/api/)
- [ROCm setup](https://zolotukhin.ai/zinc/docs/rocm/)
- [Release files and validation](https://github.com/zolotukhin/zinc/releases)
- [Report an issue](https://github.com/zolotukhin/zinc/issues)

Include `zinc --version`, your GPU/backend/runtime, model, and exact command
when reporting a problem. Remove credentials, private endpoints, and personal
paths from logs. ZINC is MIT licensed; see `LICENSE`. Models have their own
licenses and are not included in this archive.
