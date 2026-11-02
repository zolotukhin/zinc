# Gemma's Vulkan sliding-window correction

Gemma 4's sliding layers must discard keys at least 1,024 positions behind the
current query. Its global layers retain the complete causal prefix. The pinned
reference uses the [STANDARD mask](https://github.com/ggml-org/llama.cpp/blob/5f436dddb440a288ee5611d7d1eca564a6aca9f4/src/llama-hparams.h)
and [marks Gemma 4's layers explicitly](https://github.com/ggml-org/llama.cpp/blob/5f436dddb440a288ee5611d7d1eca564a6aca9f4/src/models/gemma4.cpp).

Vulkan attention previously read the whole prefix for both layer types. Short
reference continuations passed because they never crossed the sliding window.
Actual Intel OpenCode sessions with approximately 6K-token prompts exposed
repetitive output and no tool calls, even after the cache-allocation fix let
the models load at larger contexts. A successful client exit with unchanged,
failing acceptance tests remains a failed coding task.

## Correct bounds for every query

The window includes the query token. With a 1,024-token window, causal lengths
1,023 and 1,024 start at key zero; length 1,025 starts at key one. A prefill
batch crossing that boundary needs a different lower bound for each query.
The bound applies to logical positions before looking up physical KV pages.

The Vulkan push constants now carry `sliding_window_size`. Regular attention,
split attention and F32/F16 batched attention honor it. A zero window preserves
full history. Gemma 4's smaller-head sliding layers receive its model window;
the global layers receive zero, matching the existing per-layer RoPE choice.
Optimized readers that do not implement the mask remain available for global
attention. Sliding layers use the regular readers; optional Q8 caching uses
serial prefill and the window-aware Q8 decode shader.

Dense Gemma also keeps F32 activations during batched prefill by default.
A control on the extended copper prompt found an unqualified continuation
change with the extra Q8 activation approximation. Disabling that prefill
approximation restored the exact continuation; disabling decode quantization
alone did not. `ZINC_GEMMA_PREFILL_DP4A=1` explicitly opts into approximate
prefill, with a diagnostic warning. Weight quantization and F16 KV caching
are unchanged.

## Tests that cross the boundary

The numerical GPU validator compares every output element with an independent
CPU softmax oracle. Its **256 cases** cover 256/512-element heads, causal lengths
1,023/1,024/1,025/1,041, windows zero/one/17/1,024, permuted 16-token pages,
F32/F16/Q8 cache storage, regular and split attention, batched queries, output
guards and replay. The old shipped shaders fail the negative control on the
R9700; the corrected shaders pass all 256 cases on that board.

Both Vulkan release roles must run those cases in addition to the existing
84 cache-boundary and 36 Gemma-router cases: **376 numerical cases** in total.

Model qualification also includes three frozen reference continuations with
prompts beyond each Gemma's policy window, through both CLI and HTTP. Reports
retain measured prompt-token bounds and the existing reference-agreement
policy. Aggregation rejects missing, failed or shorter-than-window evidence.
These checks supplement the short continuations and request-recovery checks.

The [Intel OpenCode investigation](INTEL_OPENCODE_VALIDATION.md) records actual
coding outcomes and their source-specific scope. Numerical agreement alone
does not establish that a model completes a coding task.
