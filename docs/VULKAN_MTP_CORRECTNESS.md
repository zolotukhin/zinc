# Vulkan MTP correctness

The seven-model Intel package check found a Qwen 3.8 27B disagreement that the
earlier six-model Vulkan release role could not detect. The packaged CLI takes
a different token at position six of the lighthouse continuation. That choice
is outside the unchanged native-reference near-tie policy.

The same package with `ZINC_MTP=0` and its normal F16 cache matches all 48 native
choices across the lighthouse, refrigerator and copper prompts. Switching only
to F32 KV, disabling column matmul, or disabling chained submissions preserves
the failing MTP continuations. These are separate controls against the same
frozen native F32 references, weights and executable.

Vulkan now uses ordinary greedy decode by default. A compatible NextN block
alone does not enable speculation or allocate its draft/cache budget.
`ZINC_MTP=1` explicitly enables the experimental path and logs its correctness
limit. This change does not alter the separately validated ROCm MTP default.

CPU regressions cover absent/empty/disabled overrides, explicit opt-in and
ineligible models. Both Linux Vulkan release roles now include the pinned
NextN-bearing Qwen 3.8 model, so their normal CLI/HTTP reference gates exercise
its actual default. The failed package and precision controls remain evidence;
corrected-default package validation and real OpenCode results are separate
checks, without relaxed tolerances or substituted model responses.
