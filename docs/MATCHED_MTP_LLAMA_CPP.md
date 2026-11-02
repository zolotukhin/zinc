# Matched MTP against current llama.cpp

The November 2, 2026 UTC campaign measures **MTP on/on and off/off** on the same Radeon AI PRO R9700 and Qwen 3.8 27B Q4_K_M file. The earlier MTP-on view paired ZINC speculation with a non-speculative llama.cpp baseline. Those historical samples remain available; unequal or unknown MTP settings no longer produce a ratio.

On Vulkan, the three workloads that preserve each engine's greedy output across MTP modes show **16.6–19.6% higher ZINC decode throughput** than llama.cpp MTP. The remaining Vulkan workload and every ROCm MTP-on workload fail the exact-output control and remain diagnostic. These controls do not establish answer quality or release qualification.

## Frozen inputs

- llama.cpp: [`785322d17f2d51e5185cf996a1f4e46d87640841`](https://github.com/ggml-org/llama.cpp/commit/785322d17f2d51e5185cf996a1f4e46d87640841), upstream HEAD verified before the campaign; build 12033, `0.8.0-dev`. Both native backends were freshly built from this revision.
- ZINC: [`34f33b900969`](https://github.com/zolotukhin/zinc/commit/34f33b900969), Zig 0.17.0, ReleaseFast. The compute implementation is frozen for both backends. Later OpenCode launcher changes are outside these measured binaries.
- Measurement harness: [`df2ab9cf`](https://github.com/zolotukhin/zinc/commit/df2ab9cf). Source, executable, tokenizer and shared-library hashes are recorded independently and checked before publication.
- GPU: Radeon AI PRO R9700 (`gfx1201`), 32 GB VRAM. One GPU server at a time; no clock or shared-service changes.
- GGUF: 17,106,775,008 bytes; SHA256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Both engines: F32 target and draft K/V, 4,096 context tokens, one slot, temperature 0, seed 0, `top_p=1`, thinking off, full prompt prefill and identical output caps.
- MTP: three drafts maximum. llama.cpp uses `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 --spec-draft-sampling greedy`; off is explicit `--spec-type none`. Its default backend draft sampling remains enabled. ZINC uses `ZINC_MTP=1/0` and `ZINC_MTP_DRAFTS=3`.
- ZINC restricts its draft head to the default 98,304 vocabulary rows; llama.cpp retains its default draft vocabulary. Both use the full target model for verification. This implementation difference is disclosed, not a disabled baseline feature.

The four prompts contain 58, 183, 349 and 80 tokens respectively, identically counted by both engines. Output caps are 96, 160, 128 and 256. Three rounds reverse treatment order in the middle round. Each treatment has one warmup and three measured requests per workload: **nine measured samples per cell**, 144 measurements and 48 warmups per backend, **384 HTTP requests total**. Every request reached its common output budget and every MTP-on request recorded positive draft counts. All repeats within each engine/mode were deterministic.

## Vulkan MTP on/on

Decode rates are medians in tokens/second. HTTP times are medians in milliseconds, measured on the worker without SSH round-trip latency.

| Workload          | ZINC decode | llama.cpp decode | ZINC HTTP | llama.cpp HTTP | Exact-output control          |
| ----------------- | ----------: | ---------------: | --------: | -------------: | ----------------------------- |
| Quick chat        |       60.91 |            52.13 |   1749.35 |        2301.23 | Pass within each engine       |
| Coding review     |       78.43 |            67.25 |   2421.92 |        3060.82 | Pass within each engine       |
| Incident context  |       62.51 |            59.90 |   2764.47 |        3161.09 | llama.cpp differs; diagnostic |
| Long coding draft |       74.33 |            62.14 |   3710.66 |        4668.06 | Pass within each engine       |

All nine passing workload/round decode comparisons favor ZINC. Across those three workloads, full HTTP latency is 20.5–24.0% lower. The incident-context figures are retained without a qualified comparison ratio.

## ROCm MTP on/on: diagnostic

ZINC changes its complete greedy text between MTP off and on on all four prompts. llama.cpp changes it on quick chat. Repeated requests within each mode remain stable. These rates are measured observations, **not qualified ROCm MTP performance wins**.

| Workload          | ZINC decode | llama.cpp decode | ZINC HTTP | llama.cpp HTTP | Exact-output control |
| ----------------- | ----------: | ---------------: | --------: | -------------: | -------------------- |
| Quick chat        |       63.85 |            45.24 |   1621.83 |        2375.88 | Both engines differ  |
| Coding review     |       74.19 |            58.30 |   2455.30 |        3119.08 | ZINC differs         |
| Incident context  |       67.15 |            49.42 |   2392.64 |        3080.63 | ZINC differs         |
| Long coding draft |       72.77 |            55.08 |   3667.41 |        4937.97 | ZINC differs         |

The ROCm server uses a single-sequence path for MTP and the slot path for ordinary serving. Those paths can use different numerical kernels. That is a concrete difference to investigate; it does not by itself establish the cause or which response is more accurate. For example, quick chat changes after the initial “The ”, whereas the coding-review responses first differ at character 446. A follow-up should compare prefill and verifier logits, near-tie margins and recurrent state at the first divergent token.

Cross-engine full-text agreement is recorded separately: Vulkan agrees on three of four off-mode prompts and two of four on-mode prompts; ROCm agrees on none. Equal token counts and deterministic repeats do not replace a native-reference accuracy evaluation. No stable release or model qualification is granted by this campaign.

## MTP off/off control

These cells show ZINC / llama.cpp decode medians in tokens/second. They do not borrow either engine's MTP-on measurement.

| Workload          | Vulkan ZINC / llama.cpp | ROCm ZINC / llama.cpp |
| ----------------- | ----------------------: | --------------------: |
| Quick chat        |           34.13 / 30.59 |         32.92 / 30.35 |
| Coding review     |           33.91 / 30.52 |         32.64 / 30.17 |
| Incident context  |           33.87 / 30.47 |         32.57 / 30.04 |
| Long coding draft |           33.83 / 30.53 |         32.62 / 30.23 |

## Two measurement bugs exposed

Vulkan acceptance counters used to be logged when the _next_ request began. They could be attributed to the wrong request, and the first MTP request had no acceptance log at all. ZINC now logs the counters immediately after the request's generated-token timing, for JSON and SSE. The strict harness requires that request's positive draft telemetry.

With MTP on, ZINC's reported prefill timer stops before explicit draft priming. llama.cpp begins speculation before finalizing its prompt timer, so that work is included there. Comparing these prefill rates, or summing them with decode time to claim an overall advantage, mixes timer scopes. The dashboard therefore excludes those ratios and points readers to full HTTP latency. The raw phase timings remain unchanged.

The inspected paths are ZINC [Vulkan request timing](https://github.com/zolotukhin/zinc/blob/34f33b900969/src/server/routes.zig), [ROCm request timing](https://github.com/zolotukhin/zinc/blob/34f33b900969/src/server/cuda_serve.zig), and llama.cpp [post-decode prompt finalization](https://github.com/ggml-org/llama.cpp/blob/785322d17f2d51e5185cf996a1f4e46d87640841/tools/server/server-context.cpp).

## Evidence and reproduction

- [Vulkan complete campaign JSON](/zinc/benchmarks/2026-11-02-matched-mtp-vulkan.json)
- [ROCm complete campaign JSON](/zinc/benchmarks/2026-11-02-matched-mtp-rocm.json)
- [Interactive benchmark dashboard](https://zolotukhin.ai/zinc/benchmarks/)
- [Build and run the controlled protocol](BENCHMARKING.md#latest-llamacpp-and-matched-mtp-pairs)

The default runner fails on an exact-output difference. This investigation uses its explicit `--record-output-drift` mode: both reports have `protocol_passed: true` and `passed: false`, with the failed controls retained. Publication validates every workload hash, warmup/run count, output cap, draft count, reference revision and frozen harness hash. Its explicit diagnostic inclusion keeps failed on-mode rows excluded while preserving valid off-mode rows.

```bash
bun tools/merge_matched_mtp.mjs \
  --report <vulkan-report.json> --report <rocm-report.json> \
  --harness-commit <full-frozen-harness-commit> --include-diagnostics
```

Raw reports contain the original timings; the published view annotates phase scopes and output controls without changing samples. Replaced 27B measurements remain under each model’s `measurement_history`, with their original dates and source revisions. Source, binary and library snapshots stayed unchanged during each campaign. Every owned server was reaped before its supervising command returned.
