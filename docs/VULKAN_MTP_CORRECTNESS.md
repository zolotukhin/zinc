# Intel Vulkan MTP correctness

The seven-model Intel package check found a Qwen 3.8 27B disagreement that the
earlier six-model Vulkan release role could not detect. The packaged CLI takes
a different seventh token (index six) of the lighthouse continuation. That choice
is outside the unchanged native-reference near-tie policy.

The same package with `ZINC_MTP=0` and its normal F16 cache matches all 48 native
choices across the lighthouse, refrigerator and copper prompts. Switching only
to F32 KV, disabling column matmul, or disabling chained submissions preserves
the failing MTP continuations. These are separate controls against the same
frozen native F32 references, weights and executable.

Intel Vulkan now uses ordinary greedy decode by default. A compatible NextN
block alone does not enable speculation or allocate its draft/cache budget.
`ZINC_MTP=1` explicitly enables the experimental Intel path and logs its
correctness limit. A separate R9700 Vulkan control matches all 48 native choices
with MTP both on and off, so its existing default is preserved. This change
also retains the separately validated ROCm MTP default.

The retained [Intel precision and speculation controls](/zinc/validation/intel-qwen38-mtp-controls-2026-11-02.json)
and [R9700 control](/zinc/validation/rdna4-qwen38-mtp-controls-2026-11-02.json)
record the exact package, model pins and native-reference identities. The
[Intel OpenCode report](INTEL_OPENCODE_VALIDATION.md) keeps the corrected
default-package checks separate from coding outcomes.

CPU regressions cover absent/empty/disabled overrides, explicit opt-in and
ineligible models. Both Linux Vulkan release roles now include the pinned
NextN-bearing Qwen 3.8 model, so their normal CLI/HTTP reference gates exercise
its actual default. The failed package and precision controls remain evidence;
corrected-default package validation and real OpenCode results are separate
checks, without relaxed tolerances or substituted model responses.
