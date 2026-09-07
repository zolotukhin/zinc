# Effort 32 — Qwen 3.8 27B decode on RDNA4 (Vulkan, R9700) — STATUS: +2.7% LANDED, REMAINING HEADROOM IS THE SSM DELTA-NET STATE TRAFFIC

Model `qwen38-27b-q4k-m` (Qwen3.8-27B Q4_K_M, `qwen35` dense hybrid: 64 layers,
48 DeltaNet SSM layers + 16 full-attention layers, dense FFN 5120→17408,
lm-head Q6_K 248320×5120). Per decoded token the GPU streams ~16.1 GB of
weights plus ~0.3 GB of SSM state (48 layers × 3 MB read + 3 MB write).

Node: `hft` (192.168.0.34:2223), Radeon AI PRO R9700 (RADV GFX1201, Mesa
25.0.7), kernel 7.2.2. After the last reboot the discrete card is **Vulkan
device 0** (`vulkaninfo --summary`: GPU0 = R9700, GPU1 = Raphael iGPU), so
every run below passes `-d 0` / `--rdna-vk-device 0`. The card's sysfs
`gpu_busy_percent` / `pp_dpm_sclk` read 0% / 600 MHz even mid-decode on this
kernel — do not use them to reason about clocks; pinning
`power_dpm_force_performance_level=high` changed nothing (31.95 vs 31.95
ms/tok), so DPM is not a lever here either.

## Baseline (2026-09-03, main `4e25164b`)

Suite server path (`tools/performance_suite.mjs --target rdna`, 96 greedy tokens,
same prompt as the published site row):

| path | prefill tok/s | decode tok/s |
|---|---:|---:|
| ZINC Vulkan, core (67-tok chat prompt) | 186.6 | **31.30** (32.0 ms/tok) |
| ZINC Vulkan, decode-extended (256 tok) | 249 | 31.06 |
| ZINC ROCm, core | 392 | 32.40 |
| llama.cpp 9400c8946 llama-server (Vulkan0), core | 147 | 29.85 |
| llama.cpp 9400c8946 `llama-bench -dev Vulkan0 tg96` | — | 30.10 |

The published "48 tok/s" for this row is the suite's *end-to-end* figure
(prompt + generated tokens over total wall-clock), not decode.

Two harness gotchas found on the way:

- The `llama-server` on the node's `PATH` (b96806d96) rejects `--device
  Vulkan0`; the suite picks it first (`which llama-server`) and the whole
  baseline column comes back `unavailable_reason`. Pass
  `--rdna-llama-server /root/llama.cpp/build/bin/llama-server` (9400c8946).
- The `zinc` CLI decodes this model at 28.3 tok/s while the server path does
  31.3 tok/s on the same binary (`--chat --prompt`, 48-token prompt vs the
  server's 67). Not investigated; all accept/reject decisions here use the
  server path (`/root/server_ab.sh`: launch `zinc --port`, 1 warm-up + 3
  measured `/v1/chat/completions` calls, `max_tokens` 96).

## Method

- `--profile` per-phase GPU timestamps (inflated ~10% by the serialization
  they add; relative deltas are reliable).
- Server-path A/B with temporary env knobs, 96-token runs only. Runs shorter
  than ~60 tokens read ~0.8 ms/tok faster (early-burst boost), so the early
  30-token ablations over-stated the delta-net cost.
- Dummy-dispatch slope: +48/+96/+192 tiny dispatch+barrier pairs per token
  cost 2.5 µs each → dispatch overhead is *not* a lever on RADV for this
  model (~675 dispatches/token ≈ 1.7 ms, and most cannot be removed).
- In-place kernel ablations (skip conv / delta / gnorm; delta with loads,
  stores or reductions disabled) and cold-state re-dispatch slopes.

Per-token profile at baseline (ms, profiled): dense_ffn 18.7 (gateup 10.9 at
~594 GB/s, down 7.5 at 494/551 GB/s for Q4_K/Q6_K), ssm 12.0 (qkv+z 4.3 at
~597 GB/s combined, delta 3.2, out 2.1 at 485 GB/s, norm_ab 1.1, conv 0.4,
gnorm 0.3), attention 2.5, tail 1.7 (lm-head 628 GB/s).

## Landed (server path, 96 tokens, identical greedy output)

| change | ms/tok | tok/s |
|---|---:|---:|
| baseline | 31.95 | 31.3 |
| K-split fused RMS+alpha/beta (`rms_norm_dmmv_alpha_beta_ksplit`, k=10) | 31.6 | 31.65 |
| + NUM_ROWS=4 Q4_K/Q6_K DMMV for M == hidden_dim (`dmmv_q4k_rows4`, `dmmv_q6k_rows4`) | **31.15** | **32.1** |

- **K-split alpha/beta.** The fused RMS+alpha+beta dispatch launched
  `dt_rank` = 48 wave64 workgroups, each re-deriving the RMS and streaming
  ~100 KB (alpha row + beta row + hidden + norm weights) alone → ~24 µs/layer,
  1.13 ms/token profiled. The new shader uses a (48 × k_split) grid: every
  workgroup computes the full-vector RMS (L2-resident 20 KB re-read), row-0
  workgroups write their chunk of the normalized hidden vector, and each
  workgroup writes one partial alpha/beta dot laid out `[chunk][head]` into
  two dedicated small buffers. `ssm_delta_net_cols8` sums the `ab_ksplit`
  partials in fixed order (new trailing push-constant field, default 1), so
  results stay deterministic. Profiled norm_ab 1.13 → 0.53 ms. k=5/10/16 are
  equivalent; default 10 (`ZINC_SSM_AB_KSPLIT`, 1 disables). Decode-only:
  prefill capture/validation paths keep the plain `[dt_rank]` layout.
- **NUM_ROWS=4 for the M=5120 projections** (FFN down Q4_K/Q6_K, attention
  o-proj Q4_K): the 2-row kernels launch only 2560 workgroups and each
  re-reads the whole activation vector (68 KB for down) from L2; 4 rows per
  workgroup halves that. Profiled down 7.53 → 7.09 ms, o-proj 0.64 → 0.59.
  Gated to `isQwen36DenseHybrid27B() and M == hidden_dim` (`ZINC_DENSE_ROWS4=0`
  disables). Measured windows: M=5120 only is best; extending to z (6144)
  or qkv (10240) / attn_q (12288) is flat-to-worse; the existing 8-row
  `dmmv_q6k_wide` on the same shapes is worse than 4 rows; a Q5_K 4-row
  variant for `ssm_out` is *slower* than its 2-row kernel (rejected, not kept).

## Rejected on this exact target (do not repeat without a new hypothesis)

| attempt | result |
|---|---|
| Per-head delta-net kernel (256 threads, whole 128×128 state in registers, coalesced 256 B rows, one `subgroupAdd` per row) | correct but +0.4 ms/tok vs cols8; in-context the kernel is bound by its 6 MB/layer of cold state traffic, not by latency or reductions |
| Per-wave delta-net kernel (64 threads per 32 rows, no LDS/barriers) | same as per-head (32.4 ms/tok) |
| Pooling all 64 layers' SSM state into one 192 MB allocation (TLB / small-page hypothesis) | no change on the server path |
| DPM `high` on the R9700 | no change |
| Dispatching the z projection next to delta-net (after conv, or after delta) so the two overlap | after-conv identical, after-delta −0.8% |
| NUM_ROWS=2 / 4 for the fused gate+up+SwiGLU (spec-constant variants of the row1 kernel) | 31.2 / 31.55 vs 31.1 ms/tok for row1 — row1 stays |
| NUM_ROWS=4 on qkv (M=10240) alone | flat |

Delta-net cost anatomy (per-head kernel, in context, 96 tokens): store ≈
0.5 ms/tok, load ≈ 0.4 ms/tok, reductions ≈ 0.1 ms, plus ~0.7 ms/tok that
remains with a no-op body but disappears when the dispatch is skipped
entirely (the skip breaks the recurrence, so this last figure may include
downstream effects). Warm re-dispatch of the same layer's state costs only
6 µs (cols8) / 13 µs (per-head); against a cold buffer 17.5 / 22 µs. The
state traffic itself (3 MB read + 3 MB written back per layer) is inherent
at f32; halving it would need an f16/bf16 state, which was not attempted
because of the recurrence's precision sensitivity.

## Remaining headroom (estimates)

- delta-net state traffic ≈ 0.8–1.5 ms/tok (f16 state, or nothing);
- conv1d 0.45 ms/tok (latency-bound 160-workgroup elementwise kernel):
  fusing it into delta-net needs a 4-slot (or double-buffered) conv ring
  because 3 heads share each q/k group's state — touches the batched prefill
  conv too;
- SSM out (Q5_K, 485 GB/s) 2.1 ms/tok: a Q5_K×Q8_1 DP4a matvec is the only
  untested lever, net gain bounded by the quantize pre-pass (~0.15 ms/tok);
- everything else streams at ~590–630 GB/s already.
