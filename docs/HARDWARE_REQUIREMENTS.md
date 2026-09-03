# ZINC hardware requirements

ZINC runs on consumer GPUs through Vulkan, ROCm/HIP, Metal, and an experimental CUDA backend. This page covers the hardware, operating system, and driver stack needed for each path.

## Supported platforms

| Platform | GPU | Backend | Status |
|----------|-----|---------|--------|
| **Linux** | AMD RDNA4 discrete (Navi 48 / Navi 44) | Vulkan 1.3 | Supported and tuned |
| **Linux** | AMD RDNA4 discrete (`gfx1201`) | ROCm/HIP | Supported; R9700 is the validated reference |
| **Linux** | AMD RDNA4 APU (Strix Halo / gfx1151) | Vulkan 1.3 | Supported with APU-specific bandwidth tuning |
| **Linux** | AMD RDNA3 | Vulkan 1.3 | Supported, less tuned |
| **Linux** | Intel Arc Xe2 / Battlemage | Vulkan 1.3+ | Supported, validated benchmark target |
| **macOS** | Apple Silicon M1 through M5 | Metal | Supported, native MSL shaders |
| **Linux / WSL2** | NVIDIA RTX 40/50 series | CUDA | Experimental |

On AMD Radeon, Vulkan and ROCm are separate first-class builds. Vulkan is the broad RDNA3/RDNA4 path; ROCm uses HIP and dedicated AMD kernels, with the current public validation centered on `gfx1201`. Benchmark results are kept separate because the runtimes and driver stacks are different.

## AMD GPUs (Linux)

ZINC supports AMD consumer and workstation GPUs through Vulkan and, on validated RDNA4 stacks, ROCm/HIP.

| Family | Examples | Notes |
| --- | --- | --- |
| RDNA4 discrete (Navi 48 / gfx1201) | RX 9070, RX 9070 XT, RX 9070 GRE, Radeon AI PRO R9700 | Primary tuning target, hand-tuned shaders |
| RDNA4 discrete (Navi 44 / gfx1200) | RX 9060, RX 9060 XT | Same RDNA4 ISA as Navi 48, smaller die, narrower bus |
| RDNA4 APU (gfx1151) | Strix Halo: Radeon 8060S, Radeon 8050S | Unified-memory iGPU; ZINC selects an APU bandwidth profile (~256 GB/s) distinct from the discrete 576–640 GB/s default |
| RDNA3 | RX 7900 XTX, RX 7900 XT, RX 7800 XT, RX 7700 XT, RX 7600 | Supported, less tuned than RDNA4 |

Any AMD GPU with Vulkan 1.3 and a working RADV or AMDVLK driver should work through the Vulkan build. ROCm support is narrower and should be checked against the installed ROCm release and GPU target.

### AMD Vulkan requirements

- **OS**: Linux
- **API**: Vulkan 1.3
- **Driver**: RADV (Mesa) or AMDVLK
- **Shader compiler**: glslc (shaderc 2023.8, included in build)
- **Recommended**: `export RADV_PERFTEST=coop_matrix` on RDNA4

Verify your Vulkan stack:

```bash
vulkaninfo --summary
```

If that command does not show your AMD GPU, the Vulkan build will not work. A working ROCm/HIP device is checked separately with `rocminfo`.

### AMD ROCm/HIP requirements

The validated ROCm reference is a Radeon AI PRO R9700 (`gfx1201`) running Linux kernel `7.2.2` and ROCm userspace `7.2.4`. Those exact versions document the measured machine; the build itself does not require one fixed kernel version.

- **OS**: Linux
- **Runtime**: ROCm/HIP with a working ROCr device
- **Development libraries**: `amdhip64`, `hiprtc`, and `hipblas`
- **Current validated GPU target**: RDNA4 `gfx1201`

Verify the runtime and build the ROCm backend explicitly:

```bash
rocminfo
ROCM_PATH=/opt/rocm zig build -Dbackend=rocm -Doptimize=ReleaseFast
ROCR_VISIBLE_DEVICES=0 ./zig-out/bin/zinc --check
```

If `rocminfo` cannot see the GPU, or HIP reports `hipErrorNoBinaryForGpu`, first check that the running `amdgpu` driver, ROCr runtime, and ROCm userspace support the same target. See the [ROCm backend guide](/zinc/docs/rocm/) for the full setup and troubleshooting notes.

### AMD VRAM guide

| VRAM | What fits |
| --- | --- |
| 16 GB | 2B to 8B class models comfortably |
| 32 GB | 27B dense models and 35B MoE models like Qwen3.6-35B-A3B Q4_K_XL |

Exact fit depends on architecture, quantization, and context length. `--check -m <model>` prints a practical fit estimate.

## Intel Arc GPUs (Linux)

Intel Arc support is an official Linux Vulkan path. The current validated target is the Arc B-series / Battlemage line:

| Family | Examples | Notes |
| --- | --- | --- |
| Arc B-series desktop | Arc B580, Arc B570 | Best fit for 7B/8B models; B580 is the stronger consumer target |
| Arc Pro B-series | Arc Pro B70, B65, B60, B50 | Larger VRAM options for local AI; B70/B65 are the 32 GB targets |

### Intel requirements

- **OS**: Linux
- **API**: Vulkan 1.3 or newer, depending on card and driver
- **Driver**: Intel ANV / Mesa Vulkan driver
- **Platform**: UEFI with Resizable BAR enabled for benchmark-quality results

Verify the Vulkan stack:

```bash
vulkaninfo --summary
```

If that command does not show your Intel Arc GPU, ZINC will not use it.

### Intel VRAM guide

| VRAM | B-series cards | What fits |
| --- | --- | --- |
| 10-12 GB | B570, B580 | 7B/8B class models |
| 16 GB | B50 | 8B with more context; some 12B experiments |
| 24 GB | B60 | 20B class and tight larger-model experiments |
| 32 GB | B65, B70 | 27B dense and 35B MoE targets |

See [Intel GPU Reference](/zinc/docs/intel-gpu-reference/) for the full B-series card table, device IDs, memory bandwidth, Xe2 opcode notes, and ZINC tuning guidance.

The public benchmark matrix currently validates four managed catalog models on an Intel Arc BMG G31-class node. Decode and prefill beat the same-machine llama.cpp baseline on those rows. End-to-end server latency and Arc-specific tuning are still ongoing.

## Other Vulkan GPUs

| Family | Status |
| --- | --- |
| NVIDIA via Vulkan | Vulkan works, not primary target |

## NVIDIA RTX GPUs (experimental CUDA)

ZINC also has an experimental native CUDA backend for recent NVIDIA RTX cards. It is measured on RTX 5090 and RTX 4090 hardware, but it is younger than the supported Vulkan, ROCm, and Metal paths.

- **OS**: Linux or WSL2
- **Toolkit**: CUDA with Driver API, NVRTC, cuBLAS, and CUDA runtime libraries
- **Current development targets**: RTX 5090 (Blackwell) and RTX 4090 (Ada)

```bash
CUDA_HOME=/usr/local/cuda zig build -Dbackend=cuda -Doptimize=ReleaseFast
./zig-out/bin/zinc --check
```

Native Windows builds are not supported. WSL2 is an experimental CUDA deployment path, not a supported Vulkan or ROCm environment.

## Apple Silicon (macOS)

ZINC has a native Metal backend with 31 MSL compute shaders, zero-copy model loading via `newBufferWithBytesNoCopy`, and the same OpenAI-compatible API as the AMD path.

| Chip family | Metal GPU family | Status |
| --- | --- | --- |
| M1, M1 Pro, M1 Max, M1 Ultra | Apple7 | Supported |
| M2, M2 Pro, M2 Max, M2 Ultra | Apple8 | Supported |
| M3, M3 Pro, M3 Max, M3 Ultra | Apple9 | Supported |
| M4, M4 Pro, M4 Max | Apple9 | Supported |
| M5, M5 Pro, M5 Max | Apple10 | Supported (TensorOps investigation planned) |

### Apple Silicon requirements

- **OS**: macOS
- **Tools**: Xcode Command Line Tools (`xcode-select --install`)
- Native Metal runtime; no external GPU framework or Python environment required

### Apple Silicon memory guide

Apple Silicon uses unified memory shared between CPU and GPU. There is no separate "VRAM" budget.

| Unified memory | What fits |
| --- | --- |
| 8 GB | Too tight for most models |
| 16 GB | 2B models comfortably |
| 24 GB | 2B with headroom, 35B might be tight |
| 32+ GB | 27B dense and 35B MoE models like Qwen3.6-35B-A3B Q4_K_XL |
| 64+ GB (Pro/Max/Ultra) | Large models with generous context |

ZINC uses zero-copy model loading on Metal, so a 1.2 GB model file does not require an additional 1.2 GB of GPU memory. The model weights stay in place and the GPU reads from the mmap'd pages directly.

## Preflight check

Once the binary is built, verify everything works:

```bash
./zig-out/bin/zinc --check
```

On AMD RDNA4 with the Vulkan build, add the cooperative matrix flag. Do not set it for ROCm:

```bash
export RADV_PERFTEST=coop_matrix
./zig-out/bin/zinc --check
```

The check command verifies:

- GPU or accelerator detection for the selected Vulkan, ROCm, Metal, or CUDA build
- Shader or kernel assets needed by that backend
- Runtime initialization
- Model fit (when `-m <model>` or `--model-id <id>` is passed)

## Model catalog

See which models ZINC supports on your machine:

```bash
# Models that fit this machine
./zig-out/bin/zinc model list

# Full catalog including models that do not fit
./zig-out/bin/zinc model list --all
```

The catalog automatically selects the right GPU profile (`amd-rdna4-32gb`, `intel-arc`, `apple-silicon`, etc.) and shows which models are installed, active, and fit the available memory.

## System requirements (both platforms)

| Resource | Minimum | Recommended |
| --- | --- | --- |
| **CPU** | Any modern 64-bit (x86_64 or arm64) | Multi-core for serving |
| **System RAM** | 16 GB | 32 GB+ for larger models |
| **Storage** | SSD | NVMe SSD for fast model loading |

## Quick sanity check

### Linux (AMD Vulkan)

```bash
lspci | grep -i "vga\|display\|amd\|radeon"
vulkaninfo --summary
./zig-out/bin/zinc --check
```

### Linux (AMD ROCm)

```bash
lspci | grep -i "vga\|display\|amd\|radeon"
rocminfo
ROCR_VISIBLE_DEVICES=0 ./zig-out/bin/zinc --check
```

### Linux (Intel Arc)

```bash
lspci | grep -i "vga\|display\|intel\|arc"
vulkaninfo --summary
./zig-out/bin/zinc --check
```

### macOS (Apple Silicon)

```bash
system_profiler SPDisplaysDataType | head -20
./zig-out/bin/zinc --check
```

### Linux or WSL2 (NVIDIA CUDA, experimental)

```bash
nvidia-smi
./zig-out/bin/zinc --check
```

## Shortest path to success

### On Linux with an AMD GPU through Vulkan

```bash
export RADV_PERFTEST=coop_matrix
zig build -Doptimize=ReleaseFast
./zig-out/bin/zinc --check
./zig-out/bin/zinc model pull qwen35-9b-q4k-m
./zig-out/bin/zinc chat
```

Then see [RDNA4 Tuning](/zinc/docs/rdna4-tuning/) for performance work.

### On Linux with an AMD RDNA4 GPU through ROCm

```bash
ROCM_PATH=/opt/rocm zig build -Dbackend=rocm -Doptimize=ReleaseFast
rocminfo
ROCR_VISIBLE_DEVICES=0 ./zig-out/bin/zinc --check
./zig-out/bin/zinc model pull qwen35-9b-q4k-m
ROCR_VISIBLE_DEVICES=0 ./zig-out/bin/zinc chat
```

Then see the [ROCm backend guide](/zinc/docs/rocm/) for the validated stack, troubleshooting, and performance switches.

### On Linux with an Intel Arc GPU

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zinc --check
./zig-out/bin/zinc model pull qwen35-9b-q4k-m
./zig-out/bin/zinc chat
```

Then see [Intel GPU Reference](/zinc/docs/intel-gpu-reference/) for Arc B-series hardware details and current tuning notes.

### On macOS with Apple Silicon

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zinc --check
./zig-out/bin/zinc model pull qwen35-9b-q4k-m
./zig-out/bin/zinc chat
```

Then see [Apple Silicon Reference](/zinc/docs/apple-silicon-reference/) and [Apple Metal Reference](/zinc/docs/apple-metal-reference/) for platform details.
