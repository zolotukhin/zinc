# ZINC hardware requirements

ZINC runs on AMD GPUs (Linux, Vulkan) and Apple Silicon (macOS, Metal). This page covers what hardware and OS setup you need for each platform.

## Supported platforms

| Platform | GPU | Backend | Status |
|----------|-----|---------|--------|
| **Linux** | AMD RDNA4 | Vulkan 1.3 | Primary tuning target |
| **Linux** | AMD RDNA3 | Vulkan 1.3 | Supported, less tuned |
| **macOS** | Apple Silicon M1 through M5 | Metal | Supported, native MSL shaders |

## AMD GPUs (Linux)

ZINC targets AMD consumer and workstation GPUs that the ROCm stack does not support.

| Family | Examples | Notes |
| --- | --- | --- |
| RDNA4 | RX 9070, RX 9070 XT, Radeon AI PRO R9700 | Primary tuning target, hand-tuned shaders |
| RDNA3 | RX 7900 XTX, RX 7900 XT, RX 7800 XT, RX 7700 XT, RX 7600 | Supported, less tuned than RDNA4 |

Any AMD GPU with Vulkan 1.3 and a working RADV or AMDVLK driver should work.

### AMD requirements

- **OS**: Linux
- **API**: Vulkan 1.3
- **Driver**: RADV (Mesa) or AMDVLK
- **Shader compiler**: glslc (shaderc 2023.8, included in build)
- **Recommended**: `export RADV_PERFTEST=coop_matrix` on RDNA4

Verify your Vulkan stack:

```bash
vulkaninfo --summary
```

If that command does not show your AMD GPU, ZINC will not work.

### AMD VRAM guide

| VRAM | What fits |
| --- | --- |
| 16 GB | 2B to 8B class models comfortably |
| 32 GB | 35B MoE models like Qwen3.5-35B-A3B Q4_K_XL |

Exact fit depends on architecture, quantization, and context length. `--check -m <model>` prints a practical fit estimate.

### Future AMD directions

| Family | Status |
| --- | --- |
| Intel Arc | Possible through Vulkan, not a primary target |
| NVIDIA via Vulkan | Vulkan works, not primary target |

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
- No Vulkan, no ROCm, no MLX, no Python needed

### Apple Silicon memory guide

Apple Silicon uses unified memory shared between CPU and GPU. There is no separate "VRAM" budget.

| Unified memory | What fits |
| --- | --- |
| 8 GB | Too tight for most models |
| 16 GB | 2B models comfortably |
| 24 GB | 2B with headroom, 35B might be tight |
| 32+ GB | 35B MoE models like Qwen3.5-35B-A3B Q4_K_XL |
| 64+ GB (Pro/Max/Ultra) | Large models with generous context |

ZINC uses zero-copy model loading on Metal, so a 1.2 GB model file does not require an additional 1.2 GB of GPU memory. The model weights stay in place and the GPU reads from the mmap'd pages directly.

## Preflight check

Once the binary is built, verify everything works:

```bash
./zig-out/bin/zinc --check
```

On AMD Linux, add the cooperative matrix flag:

```bash
export RADV_PERFTEST=coop_matrix
./zig-out/bin/zinc --check
```

The check command verifies:

- GPU detection (Vulkan device or Metal device)
- Shader assets (SPIR-V on Linux, MSL sources on macOS)
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

The catalog automatically selects the right GPU profile (`amd-rdna4-32gb`, `apple-silicon`, etc.) and shows which models are installed, active, and fit the available memory.

## System requirements (both platforms)

| Resource | Minimum | Recommended |
| --- | --- | --- |
| **CPU** | Any modern 64-bit (x86_64 or arm64) | Multi-core for serving |
| **System RAM** | 16 GB | 32 GB+ for larger models |
| **Storage** | SSD | NVMe SSD for fast model loading |

## Quick sanity check

### Linux (AMD)

```bash
lspci | grep -i "vga\|display\|amd\|radeon"
vulkaninfo --summary
./zig-out/bin/zinc --check
```

### macOS (Apple Silicon)

```bash
system_profiler SPDisplaysDataType | head -20
./zig-out/bin/zinc --check
```

## Shortest path to success

### On Linux with an AMD GPU

```bash
export RADV_PERFTEST=coop_matrix
zig build -Doptimize=ReleaseFast
./zig-out/bin/zinc --check
./zig-out/bin/zinc model pull qwen3-8b-q4k-m
./zig-out/bin/zinc chat
```

Then see [RDNA4 Tuning](/zinc/docs/rdna4-tuning) for performance work.

### On macOS with Apple Silicon

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zinc --check
./zig-out/bin/zinc model pull qwen3-8b-q4k-m
./zig-out/bin/zinc chat
```

Then see [Apple Silicon Reference](/zinc/docs/apple-silicon-reference) and [Apple Metal Reference](/zinc/docs/apple-metal-reference) for platform details.
