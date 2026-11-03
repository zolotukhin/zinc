# Qwen3 0.6B Vulkan inference qualification

The managed `qwen3-0.6b-q8-0` entry is supported for inference on the recorded
R9700 and Intel Arc Pro B70 Vulkan profiles, at a default context of 2,048
tokens. The review uses existing exact-archive evidence. Final 0.1.0 candidate
qualification is still required after this catalog change.

The model remains excluded from chat and thinking recommendations. It completes
none of the three recorded Intel coding tasks on either ZINC or the separate
pinned native control. Qwen3.5 9B is the smallest model that passed those tasks.

## Evidence reviewed

| Vulkan GPU          | Packaged source                                                                                  | Starter reference choices | API result |
| ------------------- | ------------------------------------------------------------------------------------------------ | ------------------------- | ---------- |
| Radeon AI PRO R9700 | [`80a0a2c6`](https://github.com/zolotukhin/zinc/commit/80a0a2c6348800d40ef4b2ee6fae7af876e72b09) | 48/48 exact; no near ties | Passed     |
| Intel Arc Pro B70   | [`83b22e7b`](https://github.com/zolotukhin/zinc/commit/83b22e7b708f80bdb93d8727535ca6bd62677f6c) | 48/48 exact; no near ties | Passed     |

Both records use the immutable catalog GGUF pin, SHA256
`9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`,
639,446,688 bytes, and 2,048 context tokens for native inference checks.

The [R9700 readiness report](/zinc/validation/worker-readiness-window-gate-2026-11-02.json)
records archive `0.1.0-dev.41386646371`, validation source `7695a90e`, and
[workflow run 41390812424](https://github.com/zolotukhin/zinc/actions/runs/41390812424).
Its complete seven-model Vulkan check passes, including tokenizer/native
continuations, HTTP, cancellation/recovery and cleanup. The starter's 48 choices
match exactly. The archive and reference hashes remain in the original report.

The [Intel package matrix](/zinc/validation/intel-opencode-packaged-default-2026-11-02.json)
records archive `0.1.0-dev.41382805638`, validation source `b04cc7b4`, and the
same starter model pin. Its `native_defaults` starter row passes tokenizer,
all 48 native choices and HTTP. The actual OpenCode row passes 29/29 API probes
but 0/3 coding fixtures. Consequently the overall client matrix remains failed;
inference promotion does not change that outcome or the historical JSON.
The [separate native-client control](/zinc/validation/intel-starter-native-opencode-control-2026-11-02.json)
also completes 0/3 coding tasks. See the [Intel report](INTEL_OPENCODE_VALIDATION.md)
for task failures, protected-file checks, recovery and template limits.

## Catalog and backend scope

The entry now records `status=supported`, profiles `amd-rdna4-32gb` and
`intel-arc`, and `tested_backend=vulkan`. Its context, download, size and hash
are unchanged. `recommended_for_chat` and `thinking_stable` remain false.
No other GPU profile is added.

Backend scope matters on a board that runs both Vulkan and ROCm. ROCm's hybrid
forward path does not implement plain Qwen3; a successful Vulkan run on R9700
cannot qualify the ROCm path. Catalog listing and activation therefore require
the recorded backend as well as the GPU profile. Explicit preview activation
still requires its usual opt-in and the runtime's architecture checks.
ROCm starts with `qwen35-9b-q4k-m`.

The ROCm catalog probe now derives its AMD profile from the native gfx target
and total VRAM, with free VRAM used separately for fit. It no longer uses the
CUDA/NVIDIA profile. Known RDNA4 targets map to their existing VRAM tiers;
other AMD targets remain unqualified. AMD lists the R9700 as RDNA4 / `gfx1201`
in its [GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html).
This classification does not qualify additional cards.

GPU capability caches are separated by runtime backend and checked against a
versioned backend field. Legacy files are ignored and regenerated. Models,
active selection and download caches stay in their existing locations.

Stable publication checks the starter's explicit Vulkan scope, both selected
profiles, model pins/context and disabled chat/thinking recommendations.
Catalog review does not replace the mandatory final-candidate numerical,
native-reference, HTTP and cleanup reports. See [release engineering](RELEASING.md).
