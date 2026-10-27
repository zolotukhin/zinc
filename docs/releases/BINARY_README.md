# ZINC binary distribution

Run GGUF language models on your GPU from the command line or through an
OpenAI-compatible local API.

This archive contains one platform/backend build. `VERSION.json` records its
version, source commit, backend, CPU/OS requirements, runtime dependencies,
and signing status. Use the matching release's notes and `validation.json`
for the GPU, driver, model, and context combinations actually tested.

## Run the extracted package

Install the GPU driver and backend runtime first. Keep `bin/` and `share/`
together; the executable finds its matching shaders relative to its location.
Zig, Bun, Python, and a source checkout are not needed to run this package.

From the extracted directory:

```bash
./bin/zinc --version
./bin/zinc --help
./bin/zinc --check
```

For AMD Vulkan, set `export RADV_PERFTEST=coop_matrix` before the GPU check
or inference. ROCm additionally needs compatible HIP, hipRTC, hipBLAS, and
the HIP/rocWMMA headers used for runtime compilation.

Models are downloaded separately. Choose a GGUF and context limit qualified
for your backend in the release notes. Replace the path in this example:

```bash
./bin/zinc -m /path/to/model.gguf -c 2048 --chat --prompt "Explain a ring buffer."
```

Metal and Vulkan also provide browser chat:

```bash
./bin/zinc chat -m /path/to/model.gguf -c 2048
```

ROCm provides CLI/HTTP chat without the built-in browser UI or hot model
switching. See the backend's guide before selecting optional KV, MTP, or
concurrent serving settings.

## Install, upgrade, or get help

- [Binary installation and rollback](https://zolotukhin.ai/zinc/docs/binary-releases/)
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
