# ZINC binary releases

The first binary release is undergoing hardware qualification. There is no
published binary download yet; [build from source](/zinc/docs/getting-started/)
to use ZINC today. Published archives and their validation reports will appear
on [GitHub Releases](https://github.com/zolotukhin/zinc/releases).

## Choose an archive

| Hardware                                         | Archive suffix        | Requirements                                                                                                                                                  |
| ------------------------------------------------ | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Apple Silicon M1–M5                              | `macos-aarch64-metal` | Native ARM64 shell, macOS 15+, sufficient unified memory. Newer Macs retain their factory minimum OS.                                                         |
| Linux AMD RDNA3/RDNA4, Strix Halo, Intel Arc Xe2 | `linux-x86_64-vulkan` | x86_64 glibc 2.35+, Vulkan 1.3 loader and a compatible GPU driver.                                                                                            |
| Linux AMD `gfx1201`                              | `linux-x86_64-rocm`   | x86_64 glibc 2.39+, ROCm 7.2.4 with HIP, hipRTC, hipBLAS and the HIP/rocWMMA headers needed for runtime compilation. Ubuntu 24.04 is the reference userspace. |

Vulkan and ROCm are separate downloads. Strix Halo (`gfx1151`) is RDNA 3.5 and uses Vulkan. CUDA remains experimental and source-only. There is no 0.1.0 binary for Intel Macs, Linux ARM64, musl/Alpine or native Windows.

An OS/ABI floor describes the executable's compatibility. Read each release's `validation.json` for the GPU, driver, model and context combinations actually tested. It does not promise that every driver shipped with an older distribution works.

## Install a published version

Once the release is published, download `install.sh` from that release and
inspect it before running it. For `v0.1.0`:

```bash
curl -fL https://github.com/zolotukhin/zinc/releases/download/v0.1.0/install.sh -o install.sh
```

Then install:

```bash
ZINC_VERSION=v0.1.0 bash install.sh
```

On Linux this selects Vulkan. Choose ROCm explicitly:

```bash
ZINC_VERSION=v0.1.0 ZINC_BACKEND=rocm bash install.sh
```

The installer checks the archive and every file, verifies architecture/runtime compatibility, and runs `--version` before activating the new installation. It needs Bash, curl, tar and standard system utilities; it does not need Zig, Python or a source checkout. Install GPU drivers first. Vulkan archives contain compiled SPIR-V; users do not need `glslc`. Metal archives contain signed/notarized executables and their MSL sources.

The default command is `~/.local/bin/zinc`, pointing to the active version under `~/.local/share/zinc`. Add `~/.local/bin` to PATH if needed. `ZINC_INSTALL_DIR` and `ZINC_BIN_DIR` override those locations. Reinstalling identical bytes is safe; different bytes cannot replace an existing version directory.

Omit `ZINC_VERSION` to select the latest published stable release. Before a
stable release exists, this fails instead of installing a preview. Select a
published RC explicitly, for example `ZINC_VERSION=v0.1.0-rc.1`. An RC executable
cannot be renamed into the final release.

## Check the installation and choose a model

```bash
zinc --version
zinc --check
zinc model list --all
```

Follow the model/context table in that release's notes. The qualification policy includes a 639 MB Qwen3 0.6B Q8_0 starter for Metal/Vulkan and Qwen3.5 9B for ROCm. The small starter is experimental until hardware qualification is reviewed; stable publication is blocked until its catalog status is promoted in a new candidate. Larger Qwen and Gemma models need substantially more GPU/unified memory.

Managed downloads use immutable upstream revisions and SHA256 pins. Models are downloaded separately. On Metal, use `zinc model pull <id>` and `--model-id <id>` so models stay in the managed cache.

Metal/Vulkan include the browser chat UI. ROCm's standalone server provides CLI/HTTP chat and completion routes, without the built-in browser UI or hot model switching. Consult that release's report before relying on optional KV, MTP or concurrent serving configurations.

## Upgrade or roll back

Run the installer with the next published version. It activates an atomic symlink after verification. Restore the previous version with:

```bash
bash install.sh --rollback
```

Rollback switches the executable and matching shaders together. It does not move/delete models. Close running ZINC processes yourself before switching the executable used for subsequent launches.

## Verify a manual download

Archive names include platform and backend, for example `zinc-v0.1.0-linux-x86_64-vulkan.tar.gz`. Download the archive and `SHA256SUMS.txt` from the same release. Check the selected entry with `sha256sum` on Linux or `shasum -a 256` on macOS, then extract into its own directory and keep `bin/` and `share/` together.

Each release also publishes `release-manifest.json`, `release-policy.json`, `validation.json` and `provenance.jsonl`. For independent provenance verification with the GitHub CLI:

```bash
gh attestation verify zinc-v0.1.0-linux-x86_64-vulkan.tar.gz \
  --bundle provenance.jsonl --repo zolotukhin/zinc \
  --signer-workflow zolotukhin/zinc/.github/workflows/release-candidate.yml \
  --source-ref refs/heads/main --deny-self-hosted-runners
```

The publisher additionally binds the candidate run/workflow SHA, rechecks every required hardware report, and verifies public downloads against the approved bytes. See [release engineering](https://zolotukhin.ai/zinc/docs/releasing/) for automation and remaining qualification steps.
