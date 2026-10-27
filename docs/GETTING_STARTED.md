# Getting started with ZINC

Build ZINC, download a model, and open a local chat. The default build uses
Vulkan on Linux and Metal on Apple Silicon. AMD users can also choose the
[ROCm backend](/zinc/docs/rocm/).

For prebuilt archives and install/upgrade/rollback instructions, see
[binary releases](https://zolotukhin.ai/zinc/docs/binary-releases/). The 0.1.0 binary pipeline is
undergoing qualification; the instructions below describe a source build.

## Install dependencies

You need Git, the development files for your GPU backend, and Python 3 for the
initial compiler download. `scripts/zig.sh` downloads and verifies the pinned
**Zig 0.17.0** compiler when needed. A separate system Zig installation is
optional.

### macOS (Apple Silicon)

Install the Xcode Command Line Tools:

```bash
xcode-select --install
```

Check that `git` and `python3` are available. Metal is included with macOS;
you do not need Vulkan or ROCm.

### Linux (AMD or Intel Arc)

For the default Vulkan build, install your GPU driver and the
[LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home#linux). Its `glslc`
must support `GL_EXT_integer_dot_product`; older distribution versions may
not. Follow the SDK instructions to make its tools available in your shell.

On Debian or Ubuntu, install the remaining prerequisites:

```bash
sudo apt update
sudo apt install -y git python3 libvulkan-dev vulkan-tools
```

For **AMD ROCm**, follow the [ROCm setup guide](/zinc/docs/rocm/) instead of
installing the Vulkan build dependencies. It lists the HIP libraries and the
explicit `-Dbackend=rocm` build command.

NVIDIA CUDA is experimental on Linux or WSL2. Native Windows is not supported.
Check the [hardware requirements](/zinc/docs/hardware-requirements/) for the
validated GPU, driver, OS, and memory combinations.

## Clone and build

```bash
git clone https://github.com/zolotukhin/zinc.git
cd zinc
./scripts/zig.sh build -Doptimize=ReleaseFast
```

The executable is `./zig-out/bin/zinc`.

**AMD with Vulkan:** enable the cooperative-matrix path in this terminal
before checking the GPU or running a model. Skip this for ROCm, Intel Arc,
CUDA, and macOS.

```bash
export RADV_PERFTEST=coop_matrix
```

## Check your GPU

```bash
./zig-out/bin/zinc --check
```

Look for `READY [OK]`. The check verifies GPU detection, shader assets, and
runtime initialization. Resolve any reported setup issue before downloading
a model.

## Choose a model

Browse the managed catalog for your GPU profile:

```bash
./zig-out/bin/zinc model list
```

Check the model's memory requirements, then download one. This example uses
Qwen 3.5 9B:

```bash
./zig-out/bin/zinc model pull qwen35-9b-q4k-m
```

The download goes into ZINC's local cache and is verified against its SHA256
checksum. A model ID refers to a specific checkpoint in this catalog. GGUF is
the model file format; you can also [use a file you already have](#use-an-existing-gguf).

## Run your first prompt

```bash
./zig-out/bin/zinc --model-id qwen35-9b-q4k-m \
  --chat --prompt "What is the capital of France?"
```

`--chat` applies the model's conversation template. A coherent answer and a
generated-token timing in the logs confirm that the inference path is working.

## Start the chat UI

```bash
./zig-out/bin/zinc chat --model-id qwen35-9b-q4k-m
```

This starts the local server and opens the built-in chat in your browser.
The default address is `http://localhost:9090/`.

To connect an app, set its OpenAI-compatible API base URL to
`http://localhost:9090/v1` while the server is running. See the
[API guide](/zinc/docs/api/) for requests, SDK examples, and backend limitations.

## Use an existing GGUF

```bash
./zig-out/bin/zinc -m /path/to/model.gguf \
  --chat --prompt "Explain a ring buffer."
```

Use a checkpoint supported by the selected backend. The
[running guide](/zinc/docs/running-zinc/) covers server flags, system prompts,
context size, and additional CLI options.

## Manage cached models

```bash
./zig-out/bin/zinc model use qwen35-9b-q4k-m  # set the default
./zig-out/bin/zinc model active              # show the default
./zig-out/bin/zinc model rm qwen35-9b-q4k-m   # remove the cached file
```

## Models in current work

This list keeps the local ZINC models in one place. A model ID is present only when `zinc model pull` can install that checkpoint.

| Model                        | Runtime    | Model ID or access                                                                                                                            | Fits on                       | Current status                             |
| ---------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ------------------------------------------ |
| **Qwen 3.5 9B Q4_K_M**       | Local GGUF | `qwen35-9b-q4k-m` · [checkpoint](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)                                                              | 8+ GB VRAM or unified         | AMD Vulkan/ROCm, Intel Vulkan, Apple Metal |
| **Qwen 3.6 35B-A3B Q4_K_XL** | Local GGUF | `qwen36-35b-a3b-q4k-xl` · [checkpoint](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)                                                   | 24+ GB VRAM or unified        | AMD Vulkan/ROCm, Intel Vulkan, Apple Metal |
| **Qwen 3.8 27B Q4_K_M**      | Local GGUF | `qwen38-27b-q4k-m` · [checkpoint](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)                                                            | 24+ GB VRAM or 32+ GB unified | AMD RDNA4 ROCm and Apple Metal             |
| **Muse Glimmer 30B Q4_K_M**  | Local GGUF | Direct file · [exact 17 GB GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/blob/main/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf) | 24+ GB VRAM or unified        | AMD ROCm and Apple Metal validation target |
| **Gemma 4 26B-A4B Q4_K_M**   | Local GGUF | `gemma4-26b-a4b-q4k-m` · [checkpoint](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)                                                 | 16+ GB VRAM or unified        | AMD Vulkan/ROCm, Intel Vulkan, Apple Metal |
| **Gemma 4 31B Q4_K_M**       | Local GGUF | `gemma4-31b-q4k-m` · [checkpoint](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF)                                                         | 24+ GB VRAM or unified        | AMD Vulkan/ROCm, Intel Vulkan, Apple Metal |

## What to read next

- [Hardware requirements](/zinc/docs/hardware-requirements/): GPUs, drivers, memory, and OS support.
- [Running ZINC](/zinc/docs/running-zinc/): CLI and server configuration.
- [HTTP API](/zinc/docs/api/): connect apps and SDKs.
- [Documentation index](/zinc/docs/): GPU references, tuning, and engineering reports.
