# Effort 17 - RDNA4 Qwen 3.5 9B prefill gap

> **Status:** active · through cycle 35 · best 818 tok/s (long-draft harness, ~96% of llama there) · board core 32% of llama (176 vs 554)

Date: 2026-06-01

Target model:

- RDNA node GGUF: `/root/models/Qwen3.5-9B-Q4_K_M.gguf`
- Harness model key: `qwen359b`
- Site artifact id: `qwen35-9b-q4k-m`
- Architecture family: Qwen 3.5 dense SSM+attention hybrid (`qwen35`), not MoE.
- Primary metric: site-aligned `decode-extended` Long Coding Plan prefill tok/s.

## Why this effort exists

The latest published RDNA artifact replaced the old Qwen 3 8B row with
Qwen 3.5 9B. That surfaced the largest actionable RDNA gap in the
published benchmark matrix:

| scenario | prompt toks | ZINC prefill | llama.cpp prefill | ZINC pct | ZINC decode | llama.cpp decode |
|---|---:|---:|---:|---:|---:|---:|
| core | 36 | 100.79 | 548.94 | 18.4% | 95.39 | 85.51 |
| context-medium | 174 | 114.29 | 202.32 | 56.5% | 96.91 | 85.10 |
| context-long | 322 | 116.81 | 205.64 | 56.8% | 96.62 | 85.15 |
| decode-extended | 64 | 105.91 | 855.82 | 12.4% | 96.57 | 84.96 |

Decode is already ahead of llama.cpp on all four scenarios. The gap is
prefill only. The biggest absolute public gap is `decode-extended`
prefill: ZINC needs about `+749.91 tok/s` to match llama.cpp. Core
prefill is also far behind (`+448.15 tok/s`).

Metal has lower percentage rows in the site artifact, but this harness
is the RDNA remote loop. This effort therefore targets the largest
RDNA gap the current loop can actually measure and improve.

## Hypothesis

Qwen 3.5 9B is in the same dense SSM+attention architecture family as
the Qwen 3.6 dense-hybrid work, but the accepted Qwen 3.6 layer-major
prefill machinery is shape-locked to the 27B model:

```zig
fn isQwen36DenseHybrid27B(self: *const InferenceEngine) bool {
    const cfg = self.model.config;
    return cfg.n_experts == 0 and
        cfg.ssm_d_inner > 0 and
        cfg.hidden_dim == 5120 and
        cfg.intermediate_dim == 17408 and
        cfg.n_layers > 4;
}
```

That means the 9B model likely falls back to token-major `prefillBatch`
for the SSM hybrid path, or misses the important layer-major prefix and
segment paths even when individual batched kernels are available. A
5-8x prefill gap is too large for single-shader cleanup. The likely
unlock is to generalize the validated layer-major SSM+dense prefill
path to Qwen 3.5 9B with shape checks and validation.

Do not simply remove the `cfg.ssm_d_inner > 0` guard. Earlier SSM
batched-prefill work showed real dependency hazards: hidden state,
conv state, delta recurrence, residual adds, and per-layer FFN ordering
must match the token-major path.

## Measurement contract

The controller benchmark is the public Long Coding Plan prompt in raw
mode:

```text
Write an implementation plan for adding a stable benchmark preset to a
local LLM CLI. Include the command shape, warmup policy, metrics to
collect, failure handling, llama.cpp comparison, and how the site should
display prefill, decode, latency, and overall prompt+decode throughput.

Plan:
1.
```

Run shape:

- Model: `/root/models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt mode: raw
- Primary metric: ZINC prefill tok/s
- Generation cap in loop: 8 tokens, because prefill is the metric.
- Published comparison cap: 256 tokens. The prompt text is the same;
  reducing generation keeps the controller focused on prefill.
- llama.cpp target for primary scenario: `855.82 tok/s` prefill.

Success is not one lucky sample. A useful keep should:

1. Improve the primary prefill metric over the best accepted checkpoint.
2. Preserve coherent output on the five-model coherence sweep.
3. Preserve Qwen 3.5 decode's lead over llama.cpp.
4. Avoid a threshold that only helps 64-token prompts while regressing
   the 174-token and 322-token public scenarios.

## First-cycle requirements

Before changing kernels, establish the exact active path:

1. Run the baseline with `ZINC_PREFILL_PROFILE=1`.
2. Confirm whether `canUseBatchedPrefillRdna` rejects Qwen 3.5 because
   `cfg.ssm_d_inner > 0`.
3. Record the largest prefill phase bucket and sub-bucket.
4. Identify which Qwen 3.6 27B fast paths are disabled by
   `isQwen36DenseHybrid27B`.
5. If phase labels are missing for this model, add the missing profiling
   labels as a foundation step and measure again.

## Candidate implementation path

### Track 1 - Shape-safe predicate generalization

Replace the 27B-only predicate with a helper that describes the real
supported shape:

- `cfg.architecture == .qwen35`
- `cfg.n_experts == 0`
- `cfg.ssm_d_inner > 0`
- `cfg.full_attn_interval == 4`
- required attention, SSM, and dense FFN tensors are present
- tensor types are supported by the existing batched projection helpers
- RDNA device and required pipelines are present

Keep the existing Qwen 3.6 flags working, but add either neutral names
or Qwen 3.5-specific flags for any new risky path. Do not hide 9B
enablement behind an env var named only for 27B unless the old name is
just a compatibility alias.

### Track 2 - Validator before production

Reuse and generalize the existing Qwen 3.6 validation pattern:

- capture token-major reference tensors for one layer
- run the candidate batched projection / dense FFN path on the same
  tokens
- compare qkv/z/alpha/beta, gated norm, SSM output, dense FFN output,
  post-hidden, and final logits where applicable
- start with chunks of 16, 32, and 64 tokens

A validator that proves parity is a valid foundation keep even if the
production path stays off for that cycle.

### Track 3 - Minimal production enablement

Once validation passes, enable the smallest default-on production path
that can move the primary metric:

- layer-major batched dense FFN for the prefix SSM layers
- layer-major batched full-attention segment layers
- batched SSM projection where token-order recurrence remains exact

Do not assume the 27B thresholds transfer. Qwen 3.5 9B has smaller
matrices, so setup overhead can dominate short prompts. Measure 36,
64, 174, and 322-token shapes before committing a threshold.

### Track 4 - Bucket-specific follow-up

After a first production keep, use the phase budget:

- If dense FFN dominates, inspect gate/up/down tensor types and reuse the
  Qwen 3.6 DP4a dense machinery only when the 9B shapes match.
- If SSM projection dominates, batch qkv/z/alpha/beta before rewriting
  delta recurrence.
- If delta recurrence dominates, study the existing batched delta path
  and only then consider a scan-style rewrite.
- If overhead/submit/barrier dominates, look for whole-layer command
  consolidation. Do not do cosmetic barrier narrowing without a named
  measured cost.

## Known traps

- Do not optimize decode first; it is already ahead of llama.cpp.
- Do not optimize the old Qwen 3 8B row; it is no longer the RDNA
  published target.
- Do not target a synthetic Paris prompt.
- Do not bypass validation on SSM stateful layers.
- Do not port Qwen 3.6 constants blindly. The point is to generalize the
  dataflow, not to make a hidden_dim=5120 path compile for 9B.
- Do not add new `.comp` files without adding them to `build.zig` shader
  installation and proving a clean remote build actually loads them.
- Do not keep a flag-gated optimization without paired flag OFF/ON
  measurements in the same cycle.

## 2026-07-04 measured rejects

Fair server-side RDNA proxy runs on `qwen35-9b-q4k-m` / `decode-extended`
showed the current routing defaults are still better than the obvious
prefix/segment/projection toggles. Use these as cooldown evidence before
repeating the same sweep:

| toggle | prefill median | decode median | verdict |
|---|---:|---:|---|
| default reference | 1269.23 | 96.79 | keep current defaults |
| `ZINC_QWEN36_27B_DENSE_PREFILL_SEGMENT=0` | 1251.28 | 96.54 | reject |
| `ZINC_QWEN36_27B_DENSE_PREFILL_LAYERS=1` | 1261.29 | 96.52 | reject |
| `ZINC_QWEN36_27B_DENSE_PREFILL_LAYERS=2` | 1252.03 | 96.17 | reject |
| `ZINC_QWEN36_27B_DENSE_PREFILL_LAYERS=4` | 1249.95 | 96.68 | reject |
| `ZINC_QWEN36_27B_PREFIX_TAIL_PIPELINE=0` | 1243.31 | 96.03 | reject |
| `ZINC_QWEN36_27B_SSM_PREFILL_PROJ=qkv` | 1261.65 | 96.51 | reject |
| `ZINC_QWEN36_27B_SSM_PREFILL_PROJ=z` | 1246.89 | 96.54 | reject |
| `ZINC_QWEN36_27B_FULL_ATTN_BATCHED=0` | 1251.19 | 96.47 | reject |

The same session tested a K4096/K12288 exact BM64 gate-up/down experiment
behind new env flags; the fair server harness regressed from 1269.23 to
1250.91 prefill, so the engine change was not kept. Existing
`ZINC_QWEN35_9B_SSM_Q5_BK2` remains useful: disabling it landed at
1250.93 prefill. The remaining biggest buckets from `ZINC_PREFILL_PROFILE=1`
are dense FFN gate/up/down and SSM projection internals, not routing depth.

## Full-matrix follow-up

After any material keep above 150 tok/s on the primary metric, run or
prepare the full RDNA Qwen 3.5 matrix:

```bash
bun tools/performance_suite.mjs \
  --target rdna \
  --phase all \
  --rdna-sync \
  --rdna-build \
  --models qwen35-9b-q4k-m \
  --rdna-start-llama \
  --rdna-vk-device 1 \
  --require-rdna-device-substring GFX1201 \
  --runs 3 \
  --warmup 1
```

Do not publish a controller win until it is at least directionally sane
on the other public prompt lengths.
