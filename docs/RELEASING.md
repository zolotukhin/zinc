# Releasing ZINC

The release implementation builds three archives, certifies those exact archives on configured GPU workers, and promotes them after one protected publication approval. It does not publish on a tag push. The target, model, ABI and hardware policy is [config/release/targets.json](https://github.com/zolotukhin/zinc/blob/main/config/release/targets.json).

The first release is not published yet. Track the concrete remaining gates in
[0.1.0 readiness](https://github.com/zolotukhin/zinc/blob/main/docs/releases/0.1.0-plan.md).

## Version policy

Versions are `MAJOR.MINOR.PATCH`, tags are `vMAJOR.MINOR.PATCH`. Before 1.0, MINOR covers new features/models/backends, performance work and breaking changes; PATCH covers fixes/docs without client-visible contract changes. From 1.0, breaking changes require a MAJOR bump. Call out breaking CLI/API/cache changes in the release preparation PR.

Land implementation changes through ordinary reviewed PRs. A later release preparation PR contains the agreed version, scope/known limitations, and user-facing documentation changes. Merge it before freezing the final candidate. Actual hardware/model tables and download names are generated from candidate evidence, not hand-filled placeholders.

## Workflows

| Workflow                | Trigger                           | Result                                                                                                                                                                   |
| ----------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `release-preview.yml`   | Relevant PRs/main changes, manual | Unsigned CPU-only packages for Metal, Vulkan and ROCm, with ABI/dependency audits and actual archive installation outside the checkout. No GPU coverage or publication.  |
| `release-candidate.yml` | Manual from main; opt-in nightly  | Frozen full SHA/version, host tests, baseline builds, Developer ID/notarization, mandatory GPU reports, generated notes, attestation and a `release-candidate` artifact. |
| `release.yml`           | Manual candidate run ID from main | Read-only preflight summary, protected approval, re-verification, final tag, draft/upload/read-back/publish, then public-download and clean-install checks. No rebuild.  |

PR code never runs on GPU/signing workers. Actions use commit pins, checkout credentials are not retained, and only the publisher has release-write permission. Nightly runs cannot be promoted. Enable them with repository variable `ZINC_ENABLE_RELEASE_NIGHTLY=true` after provisioning workers and signing configuration.

The nightly schedule runs the full correctness/model matrix. Performance baselines and broader optional-feature qualifications remain separate work; this pipeline makes no TPS claims. Current checks use greedy decoding, one slot, f32 KV and the backend's default MTP setting at the policy's context limits. Metal/Vulkan include browser chat; ROCm exposes CLI/HTTP routes without the built-in UI or hot model switching. Add checks before expanding that contract.

## Build environments and package format

The repository pins Zig and Bun. Linux builds export the frozen Git tree, excluding untracked files, credentials, models and caches. Vulkan links inside a digest-pinned Ubuntu 22.04 image (glibc 2.35), using commit/hash-pinned Vulkan headers and versioned shaderc. ROCm uses a digest-pinned 7.2.4 Ubuntu 24.04 image and pinned hipBLAS/rocWMMA development packages. Both build for baseline x86_64 with explicit glibc targets.

Metal builds for baseline ARM64 with a macOS 15.0 deployment target and the hosted worker's SDK. The hosted image/SDK can change: inspect and retain its build log; the archive does not claim bit-for-bit reproducibility across SDK updates. Mach-O deployment target/system dependencies and ELF architecture/glibc/DT_NEEDED/RPATH are audited. Native `--version` must succeed before packaging and installation.

Archives contain `bin/zinc`, CPU-only `bin/zinc-tokenize`, matching runtime assets, `VERSION.json`, `INSTALL-METADATA`, `FILES.sha256`, binary usage instructions and license. The packaged README comes from `docs/releases/BINARY_README.md`, rather than the source-checkout setup guide. The build emits a complete shader inventory; missing assets fail packaging. Owners/timestamps/order are normalized. Apple timestamped signatures are final content, frozen before hardware certification. Python 3 is a development/CI tool only; the installed binary and Bash installer add no Python/Node dependency.

## Provision GPU workers

Use dedicated workers with generic role labels. Set these repository variables to JSON arrays of labels, without hostnames, addresses, serials or private machine names:

| Variable                    | Required role                | Practical allocation for this matrix               |
| --------------------------- | ---------------------------- | -------------------------------------------------- |
| `ZINC_RUNNER_METAL_APPLE7`  | M1 / Apple7, macOS 15        | 8 GB UMA; OS minimum + starter                     |
| `ZINC_RUNNER_METAL_APPLE8`  | M2 / Apple8                  | 8 GB UMA; starter                                  |
| `ZINC_RUNNER_METAL_APPLE9`  | M3 or M4 / Apple9            | Large-memory worker, 48 GB+; all listed models     |
| `ZINC_RUNNER_METAL_APPLE10` | M5 / Apple10                 | 16 GB+; starter + Qwen3.5 9B                       |
| `ZINC_RUNNER_RDNA3`         | AMD RDNA3 Vulkan             | 12 GB+; starter + Qwen3.5 9B                       |
| `ZINC_RUNNER_RDNA4`         | AMD `gfx1201`, Vulkan + ROCm | 32 GB; Ubuntu 24.04/ROCm 7.2.4; shared reservation |
| `ZINC_RUNNER_STRIX_HALO`    | RDNA 3.5 / `gfx1151` Vulkan  | Enough assigned UMA for starter + Qwen3.5 9B       |
| `ZINC_RUNNER_INTEL_XE2`     | Intel Battlemage Vulkan      | 32 GB for required dense/MoE matrix                |

A generic label array is `["self-hosted", "linux", "x64", "zinc-rdna4"]`. Missing/empty assignments fail resolution. The packaged executable verifies the actual selected family; software Vulkan implementations and device-index fallbacks fail.

Each runner service sets `ZINC_RELEASE_WORKER_CONFIG` to a private JSON file outside the checkout. Replace these placeholders locally; `device` must become an integer:

```json
{
  "device": "<GPU index as integer>",
  "lock_directory": "<absolute shared reservation directory>",
  "reference_source": "<absolute pinned llama.cpp source directory>",
  "reference_bin": "<absolute llama.cpp binary directory>",
  "reference_args": ["-ngl", "99", "--device", "<reference GPU selector>"],
  "models": {
    "<required model id>": "<absolute verified managed or canonical model path>"
  }
}
```

Populate every model assigned to that profile in the policy. On Metal, use `zinc model pull <id>` into `~/Library/Caches/zinc/models/models/<id>/model.gguf`. Configure canonical paths on existing Linux workers without relocating other users' models. Every run rechecks model size and SHA256. Qwen3.8 uses the historical revision matching its existing digest. Gemma 31B now has an exact upstream pin instead of an estimated size; old local files with different bytes need qualification against this pin before release.

For a worker serving multiple profiles, an optional `profiles` object maps profile IDs to overrides for `device`, `reference_args`, `reference_bin`, `reference_source` or `models`. Vulkan and HIP often assign different indices to the same card, so configure each backend's observed index/reference flags separately. Model-path overrides merge with the base model map. The shared `lock_directory` cannot be overridden by profile.

Build the exact llama.cpp commit in the policy. The harness checks clean Git source, executable version and hashes. Workers need Python 3, Git and the supported backend runtime. Vulkan workers also need `cc`, Vulkan headers, `vulkaninfo`, and matching shared `libggml-base` beside the reference tools for the 84-case numerical shader check.

The policy also freezes reference generation: one slot, a 512-token logical
batch, one-token microbatches, F32 K/V, and flash attention disabled. The harness
applies these settings after worker arguments and records them in the report;
aggregation rejects missing or different settings. Quantized batched and serial
paths can choose different greedy tokens within the reference engine itself.
Explicit settings make continuation comparisons reproducible. Throughput
measurements use the performance suite and its matched scenario settings.

Reference agreement must be at least 97.5% exact across each model's cases.
One differing choice per case can qualify only when it is the reference's
second choice within 0.05 log probability. The harness checks the complete
rendered prompt's tokenization, teacher-forces that choice, requires the entire
remaining suffix to match, and verifies the complete detokenized CLI text. HTTP
responses must match the original reference or that qualified continuation.
Reports retain the original reference and record every accepted choice, gap,
and exact-token count. Aggregation checks these measurements against policy.

Vulkan/ROCm jobs share workflow concurrency and the same configured host reservation. Its parent must exist. The harness refuses an occupied reservation or visible ZINC/llama GPU process; it never stops unrelated workloads. Servers belong to owned process groups and are stopped/reaped on errors, timeouts and normal cancellation before the command returns. A hard worker crash can leave a reservation directory: inspect processes before clearing it manually.

Only sanitized JSON is uploaded: source/archive/binary/model digests, public GPU/OS/driver data and results. Private paths, endpoints, process arguments, serials and UUIDs are excluded. Temporary worker logs are removed. Checks cover packaged asset discovery, diagnostics, tokenizer corpus, reference continuations, HTTP health/models/raw/chat SSE, cancellation/recovery and context boundaries. Missing, duplicate, failed or stale reports cannot certify a candidate.

## Configure Apple signing and release approval

Create `release-signing`, restricted to main, with these environment secrets:

- `ZINC_APPLE_CERTIFICATE_BASE64`: Developer ID Application certificate/private key exported as PKCS#12, base64 encoded.
- `ZINC_APPLE_CERTIFICATE_PASSWORD`: PKCS#12 password.
- `ZINC_APPLE_SIGNING_IDENTITY`: matching Developer ID Application identity.
- `ZINC_APPLE_API_KEY_BASE64`, `ZINC_APPLE_API_KEY_ID`, `ZINC_APPLE_API_ISSUER`: App Store Connect notarization API credentials.

The disposable macOS job imports into a temporary keychain, signs both executables with hardened runtime/timestamps, submits a ZIP to Apple, requires `Accepted`, then packages the signed files once. It restores the original keychain search list and deletes temporary keychain/credentials in `finally`. Hardware jobs run `codesign` and `spctl` against the package. Unsigned PR archives cannot pass candidate notarization. Command-line executables do not receive an app-bundle stapling step.

Create `release` with at least one required reviewer, restricted to main. The publisher reads its configuration and refuses a missing/unprotected environment; naming an environment in YAML is insufficient. Decide whether self-review is permitted for the maintainer setup. Preflight presents the candidate run, full source SHA and archive hashes before approval.

## Qualify, freeze and publish

After workflows merge to main and infrastructure is configured:

1. Dispatch `release-candidate.yml` on main with an explicit nightly/RC version first. Resolution accepts main or a full SHA already on main, freezes it once, and rejects dirty source/incomplete hardware assignments.
2. Review all nine reports and model results. The small starter starts experimental. Promote its catalog status/tested profiles only after reviewing real Metal/Vulkan family evidence. Stable publication rejects an unqualified starter; that source change needs another candidate.
3. Dispatch the final candidate with `version=0.1.0`. An RC cannot be renamed. Every final archive already embeds `0.1.0` and the full SHA.
4. Review the `release-candidate` artifact: three archives, policy, checksums, validation, generated notes, installer and offline provenance. Artifacts expire after 14 days; rebuild/requalify if unavailable.
5. Dispatch `release.yml` on main with the successful manual candidate run ID. Preflight verifies workflow identity, source ancestry, provenance, complete model/report matrix and protected environment. Review its summary, then approve `release`.
6. Publication re-verifies, creates the final tag at the tested SHA, creates/resumes only this candidate's draft, uploads missing assets without overwriting, downloads/verifies every byte, then publishes. Dependent jobs verify public downloads/installs without relying on token-created tag events to trigger another workflow.
7. Review all public checks and backend/model/download documentation before closing issue #10.

The same published candidate can be rechecked without writes. Different bytes or another candidate under an existing version fail. A partially uploaded draft for this candidate can resume. An orphan tag without its candidate draft requires manual inspection. Failed public checks remain visible; the workflow does not delete or silently replace a release.

## Qualification still required before 0.1.0

These jobs supply gates, not hardware or an Apple account. Provision the eight worker roles/nine profiles, Apple credentials and protected environment; run the artifact matrix; review/promote the starter; rehearse a candidate/publication. Broader optional-feature, memory-rejection and performance-regression qualification remains necessary before advertising those features. Merging implementation does not create a public 0.1.0 tag, release or issue closure.
