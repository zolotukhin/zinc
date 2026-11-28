# Install ZINC and run your first model

Install a Linux binary, download a model, and get a response. No Zig compiler,
Python environment, or source checkout is needed for the stable release.
GPU drivers and model weights are installed separately.

Already installed? Continue to [Run models and connect apps](RUNNING_ZINC.md).
Returning to maintain an installation? Jump to [upgrade and rollback](#upgrade-or-roll-back)
or [manual verification](#verify-a-manual-download).

## Choose your backend

[Stable 0.1.2](https://github.com/zolotukhin/zinc/releases/tag/v0.1.2)
provides two **Linux x86_64** downloads. Each executable contains one backend.

| Choose                                  | Qualified release hardware                       | Runtime requirements                                                      | Interfaces                  |
| --------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------- | --------------------------- |
| **HIP / ROCm** (preferred for AMD RDNA) | AMD Radeon AI PRO R9700 (`gfx1201`)              | glibc 2.39+, ROCm 7.2.4 with HIP, hipRTC, hipBLAS and HIP/rocWMMA headers | CLI, browser chat, HTTP API |
| **Vulkan** (Intel; AMD alternative)     | Intel Arc Pro B70 / Xe2; AMD Radeon AI PRO R9700 | glibc 2.35+, Vulkan 1.3 loader and compatible GPU driver                  | CLI, browser chat, HTTP API |

Vulkan archives include compiled shaders, so `glslc` is unnecessary. ROCm
compiles kernels at runtime and still needs its development libraries and
headers; use the [ROCm prerequisites](ROCM.md#runtime-prerequisites).

These are the release's tested GPU profiles, not a promise for every card with
the same architecture. Other AMD GPUs, Metal previews and experimental CUDA
have separate [hardware and release limits](HARDWARE_REQUIREMENTS.md).
There is no stable 0.1.2 archive for macOS, Linux ARM64, musl/Alpine or Windows.
Apple Silicon users should follow [preview or source installation](#build-from-source).

## Install ZINC

Download the release installer and review it:

```bash
curl -fL https://github.com/zolotukhin/zinc/releases/download/v0.1.2/install.sh -o install.sh
```

For **AMD RDNA**, select the native **HIP / ROCm** archive:

```bash
ZINC_VERSION=v0.1.2 ZINC_BACKEND=rocm bash install.sh
```

For **Intel Arc**, or the alternative **AMD Vulkan** path:

```bash
ZINC_VERSION=v0.1.2 ZINC_BACKEND=vulkan bash install.sh
```

Then make the command available in this shell:

```bash
export PATH="$HOME/.local/bin:$PATH"
zinc --version
```

Add that PATH line to your shell startup file to keep it for future terminals.
The installer needs Bash, curl, tar and standard system utilities. It checks
file integrity, architecture and runtime compatibility before activating the
installation. The version output identifies the selected backend.

Use the explicit backend above with the published 0.1.2 installer. The current
source installer selects HIP on AMD-only Linux machines and checks
runtime-compilation headers before every HIP install. It selects Vulkan on Intel or
GPU-less hosts and requires a choice on mixed-vendor machines. Rollback uses
the saved installation receipt. This selection does not install drivers or
qualify additional GPU models.

The default command is `~/.local/bin/zinc`; its matching runtime assets live
under `~/.local/share/zinc`. `ZINC_BIN_DIR` and `ZINC_INSTALL_DIR` override these
locations. Reinstalling identical bytes is safe.

## Check your GPU

For **AMD Vulkan on RDNA4**, set this in the shell that will run ZINC:

```bash
export RADV_PERFTEST=coop_matrix
```

Intel Vulkan, ROCm and Metal do not use that setting. Now check the installation:

```bash
zinc --check
```

Proceed only when the check exits successfully with a ready verdict; read
any warnings first. A failed check exits nonzero. This checks readiness,
not model output quality. If it fails, use the
[troubleshooting table](RUNNING_ZINC.md#troubleshooting) before downloading a model.

## Choose a model

```bash
zinc model list
```

This lists models tested for the detected GPU profile and estimated to fit its
memory. For the full catalog, use `zinc model list --all`; for scripts, use
`zinc model list --json`.

**Qwen3.5 9B Q4_K_M** is the starting chat model in this guide. If it appears in
your supported list, download it and check its fit at a modest 4K context:

```bash
zinc model pull qwen35-9b-q4k-m
zinc --check --model-id qwen35-9b-q4k-m -c 4096
```

Downloads use pinned upstream revisions and SHA256 checks and stay in ZINC's
managed cache. Leave space for the model and future downloads. Larger context
windows need additional memory; see [model sizing](HARDWARE_REQUIREMENTS.md#model-memory).

For a smaller **Vulkan installation test**, the 639 MB `qwen3-0.6b-q8-0` model
has [qualified coverage at 2K context](STARTER_QUALIFICATION.md). It is not a
recommended chat or agent model. Memory fit and passing a small inference test
do not establish coding ability.

## Run your first prompt

```bash
zinc --model-id qwen35-9b-q4k-m -c 4096 \
  --chat --prompt "What is the capital of France? Answer in one sentence." -n 64
```

Expect a response identifying Paris and a normal process exit. If output is
missing or corrupt, preserve the version, check output and command for
[troubleshooting](RUNNING_ZINC.md#troubleshooting).

## What to do next

- [Open browser chat](RUNNING_ZINC.md#browser-chat) on HIP, Vulkan or the Metal preview.
- [Start the API and connect an app](RUNNING_ZINC.md#start-the-api) on any backend.
- Connect [OpenClaw](OPENCLAW.md) or [OpenCode](OPENCODE.md) with their standalone helpers.
- [Choose another model or use an existing GGUF](RUNNING_ZINC.md#models-and-cache).
- [Run Inkling 2 Mini on AMD RDNA4](RUNNING_ZINC.md#inkling-2-mini-on-amd-rdna4) with a current Linux source build.

## Upgrade or roll back

Stop your running ZINC processes before switching versions. Rerun the installer
with the next published `ZINC_VERSION`; omitting it selects the latest stable
release. Explicit versions keep installations reproducible. RCs require an
explicit version and have their own validation records.

Restore the previous installation with:

```bash
bash install.sh --rollback
```

The installer switches the executable and matching runtime assets together.
Cached models stay in place. The certified 0.1.2 archives passed
[installation, upgrade and rollback checks](/zinc/validation/release-candidate-0.1.2-2026-11-27.json).

## Verify a manual download

Download the chosen archive and `SHA256SUMS.txt` from the
[same release](https://github.com/zolotukhin/zinc/releases/tag/v0.1.2).
Compare the archive's `sha256sum` with its entry, then extract it into its own
directory. Keep `bin/` and `share/` together: the Vulkan executable needs the
matching `share/zinc/shaders/` assets.

The release also publishes `release-manifest.json`, `release-policy.json`,
`validation.json` and `provenance.jsonl`. For independent provenance verification:

```bash
gh attestation verify zinc-v0.1.2-linux-x86_64-vulkan.tar.gz \
  --bundle provenance.jsonl --repo zolotukhin/zinc \
  --signer-workflow zolotukhin/zinc/.github/workflows/release-candidate.yml \
  --source-ref refs/heads/main --deny-self-hosted-runners
```

The [published validation](/zinc/validation/release-candidate-0.1.2-2026-11-27.json)
records exact GPU, driver, model and context combinations. See
[release engineering](RELEASING.md) for the publishing process.

## Build from source

Use the [Development guide](DEVELOPMENT.md#prerequisites) for prerequisites,
backend selection and the build command. It is also the path for unreleased
features and experimental CUDA. A source build produces `./zig-out/bin/zinc`;
add its `bin` directory to PATH to follow the usage guides.

Apple Silicon also has unsigned Metal archives in the
[preview workflow artifacts](https://github.com/zolotukhin/zinc/actions/workflows/release-preview.yml).
They require native ARM64, macOS 15+ and the Mac's factory OS minimum. These
previews are separate from stable Linux assets and have no notarization or
stable hardware qualification claim.
