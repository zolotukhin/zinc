# ZINC binary releases

[Linux 0.1.0-rc.1](https://github.com/zolotukhin/zinc/releases/tag/v0.1.0-rc.1)
is available as a prerelease, with Vulkan and ROCm archives, checksums, an
installer and signed provenance. All three mandatory GPU profiles pass on the
[exact candidate bytes](/zinc/validation/release-candidate-0.1.0-rc.1-2026-11-03.json).
The final `0.1.0` candidate has also passed
[qualification](/zinc/validation/release-candidate-0.1.0-2026-11-03.json) and
awaits protected publication. Select the
RC version explicitly when installing; it is not the default stable download.

## Choose an archive

| Hardware                                | Archive suffix        | Requirements                                                                                                                                                  |
| --------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linux R9700 and Intel Arc Pro B70 / Xe2 | `linux-x86_64-vulkan` | x86_64 glibc 2.35+, Vulkan 1.3 loader and a compatible GPU driver.                                                                                            |
| Linux AMD R9700 (`gfx1201`)             | `linux-x86_64-rocm`   | x86_64 glibc 2.39+, ROCm 7.2.4 with HIP, hipRTC, hipBLAS and the HIP/rocWMMA headers needed for runtime compilation. Ubuntu 24.04 is the reference userspace. |

Vulkan and ROCm are separate downloads. This is a Linux-first release; RDNA3,
Strix Halo and other GPUs remain outside its qualified launch matrix. CUDA
remains experimental and source-only. There is no stable 0.1.0 binary for Macs,
Linux ARM64, musl/Alpine or native Windows.

Metal remains an unsigned preview. Its CPU-built archives are available from
[packaging workflow artifacts](https://github.com/zolotukhin/zinc/actions/workflows/release-preview.yml),
alongside [source builds](/zinc/docs/getting-started/). They require native ARM64,
macOS 15+ and each Mac's factory OS minimum, and have no notarization or stable
hardware qualification claim. They are separate from the Linux release assets.

An OS/ABI floor describes the executable's compatibility. Read each release's `validation.json` for the GPU, driver, model and context combinations actually tested. It does not promise that every driver shipped with an older distribution works.

## Install a published version

Download `install.sh` from the published RC and inspect it before running it:

```bash
curl -fL https://github.com/zolotukhin/zinc/releases/download/v0.1.0-rc.1/install.sh -o install.sh
```

Then install:

```bash
ZINC_VERSION=v0.1.0-rc.1 bash install.sh
```

On Linux this selects Vulkan. Choose ROCm explicitly:

```bash
ZINC_VERSION=v0.1.0-rc.1 ZINC_BACKEND=rocm bash install.sh
```

The installer checks the available archive list and every file, verifies architecture/runtime compatibility, and runs `--version` before activating the new installation. It needs Bash, curl, tar and standard system utilities; it does not need Zig, Python or a source checkout. Install GPU drivers first. Vulkan archives contain compiled SPIR-V; users do not need `glslc`. A Linux-only release cleanly rejects a Mac installation request before downloading an archive.

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

Follow the model/context table in that release's notes. The 639 MB Qwen3 0.6B
Q8_0 starter has [reviewed Vulkan inference coverage](STARTER_QUALIFICATION.md)
on R9700 and Intel Arc Pro B70 at 2,048 context tokens. Use it for a small
installation check. It is not recommended for chat, reasoning or coding.
Qwen3.5 9B is the starting chat model on either Linux backend and the smallest
model that passed the recorded Intel coding tasks. Those client checks remain
separate from release qualification. Larger Qwen and Gemma models need
substantially more GPU memory. Stable publication requires all three
final-candidate profiles, including the reviewed starter scope.

Managed downloads use immutable upstream revisions and SHA256 pins. Models are downloaded separately. On Metal, use `zinc model pull <id>` and `--model-id <id>` so models stay in the managed cache.

Metal/Vulkan include the browser chat UI. ROCm's standalone server provides CLI/HTTP chat and completion routes, without the built-in browser UI or hot model switching. Consult that release's report before relying on optional KV, MTP or concurrent serving configurations.

## Upgrade or roll back

Run the installer with the next published version. It activates an atomic symlink after verification. Restore the previous version with:

```bash
bash install.sh --rollback
```

Rollback switches the executable and matching shaders together. It does not move/delete models. Close running ZINC processes yourself before switching the executable used for subsequent launches.

## Verify a manual download

Archive names include platform and backend, for example `zinc-v0.1.0-rc.1-linux-x86_64-vulkan.tar.gz`. Download the archive and `SHA256SUMS.txt` from the same release. Check the selected entry with `sha256sum` on Linux or `shasum -a 256` on macOS, then extract into its own directory and keep `bin/` and `share/` together.

Each release also publishes `release-manifest.json`, `release-policy.json`, `validation.json` and `provenance.jsonl`. For independent provenance verification with the GitHub CLI:

```bash
gh attestation verify zinc-v0.1.0-rc.1-linux-x86_64-vulkan.tar.gz \
  --bundle provenance.jsonl --repo zolotukhin/zinc \
  --signer-workflow zolotukhin/zinc/.github/workflows/release-candidate.yml \
  --source-ref refs/heads/main --deny-self-hosted-runners
```

The publisher additionally binds the candidate run/workflow SHA, rechecks every required hardware report, and verifies public downloads against the approved bytes. See [release engineering](https://zolotukhin.ai/zinc/docs/releasing/) for automation and remaining qualification steps.
