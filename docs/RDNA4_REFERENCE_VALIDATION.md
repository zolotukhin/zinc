# RDNA4 reference validation audit

Packaged checks must identify the reference executable and its libraries, as
well as its Git checkout. This audit found a stale HIP server while preparing
fresh R9700 tests. The existing version gate rejected it before model loading.
The [portable evidence](/zinc/validation/rdna4-reference-audit-2026-11-01.json)
preserves that failure, the earlier diagnostics and the reference rebuild.

## Reference identity correction

The HIP server with SHA-256
`5bc83033fc32c0752f8b00469de7b9d1ad4acb8da132fbe52ee1607e52986bbc`
reports revision `b96806d96`. Its checkout is at the intended policy revision,
`5f436dddb440a288ee5611d7d1eca564a6aca9f4`. The older reports correctly recorded
the executable hash, but their claim that this HIP executable verifies the
policy revision is withdrawn. Its recorded continuations remain diagnostics
against those exact reference bytes. They cannot qualify the intended reference.
OpenCode fixture outcomes and ZINC's own before/after comparisons are separate
observations and retain their recorded scope.

Building a missing `llama-tokenize` also replaced sibling shared libraries while
leaving that server executable unchanged. Reconfiguring and building both tools
produced a server that reports the intended revision. Both pre-build and
post-build hashes are retained; the rebuilt tools require fresh model checks.

The candidate and worker-readiness harnesses now snapshot both executables and
all sibling ggml/llama libraries before model checks. They refuse success if any
of these files changes. Published reports contain only library basenames and
hashes. Candidate aggregation requires that inventory. Static builds may have
an empty library inventory.

Installed ZINC runs with inherited model and library overrides removed.
Reference processes explicitly select their own sibling libraries. The
reference's library search path is kept out of ZINC's execution environment.

## Native reference backends

The initial test of the unsigned main preview `9eb380e0` used the pinned Vulkan
reference for both ZINC backends. Ten of twelve model/backend rows passed. Its
ROCm MoE row disagreed on “fridge” versus “box”; the Vulkan reference's logprob
gap was about 0.091, above the allowed 0.05. A CPU reference chose “box” too.
The older HIP executable chose “fridge”, and all six ROCm CLI diagnostics
matched it, but the identity correction above prevents treating those results
as policy qualification.

Both automated paths now require one explicit reference device for the tested
backend: Metal for Metal, Vulkan for Vulkan, and ROCm for ROCm. Missing,
duplicate, CPU-only and cross-backend selectors fail. Reports record the
reference backend, and aggregation verifies it. Cross-backend comparisons
remain diagnostics; the token agreement and near-tie limits are unchanged.

## Unicode and recovery probes

The initial Vulkan starter row failed the literal Unicode echo requirement.
With the generic assistant prompt, ZINC replied `☕ 东京咖啡 🚀`; llama.cpp also
rephrased the request. Increasing the output budget did not change that behavior.
Main's explicit copy-task system prompt addresses this probe while preserving
the required Latin, Japanese and four-byte characters. The original failure is
retained rather than relabelled a pass.

Fresh tests must also use main's stronger active chat-stream cancellation and
EOS checks. The earlier raw-route disconnect did not prove an active streaming
generation was cancelled. An old package's green result cannot certify changed
executables or the stronger harness.

## Regression coverage and release scope

CPU lifecycle tests exercise candidate and readiness entrypoints with a stable
reference, a stale server version, a wrong native backend, and a library-only
change. They verify failure reports and reservation cleanup without GPU work.
Separate checks cover inherited environments, process-specific library paths
and malformed aggregation evidence. Negative controls reject omitted identity
and backend checks.

All GPU activity in this audit ran on the remote R9700 with one model process
at a time. Every owned process was stopped and reaped before its command ended.
The unsigned previews and diagnostic reports do not qualify a signed candidate.
The [release readiness plan](releases/0.1.0-plan.md) tracks the current worker,
signing and nine-profile requirements.
