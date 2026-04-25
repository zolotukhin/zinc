# Effort 10 — Qwen 3.6 35B-A3B decode speedups on RDNA4

## Where the time goes

`--profile` on `qwen36-35b-a3b-q4k-xl` (R9700, 8-token prompt, 9 sampled
decode steps):

```
GPU decode token=37.86 ms  (= 26.4 tok/s)
  attention =  4.65 ms  (12%)   over 10 attention layers
  ssm       = 15.94 ms  (42%)   over 30 SSM layers
    proj    =  8.81 ms  (4 DMMVs sharing norm_buf input)
    conv    =  0.42 ms
    delta   =  3.59 ms  (ssm_delta_net)
    gnorm   =  0.34 ms
    out     =  3.10 ms
  moe       = 13.24 ms  (35%)
    router  =  2.20 ms
    topk    =  0.57 ms
    gate_up =  4.66 ms  (kpar already enabled)
    swiglu  =  0.31 ms
    down    =  4.67 ms  (kpar already enabled)
    acc     =  0.52 ms
  tail      =  0.96 ms  (LM head, vocab 248320 — wide-NUM_ROWS=8 hits)
  embed     =  0.01 ms
```

Architecture: `qwen35moe`, 40 layers, full_attn_interval=4 (so 10
attention + 30 SSM layers), 16 heads (2 KV), hidden 2048, vocab 248320,
n_experts=128, n_experts_used=8, Q4_K_XL.

Active params per token ≈ 3 GB (3B params × 0.65 B/param Q4_K_XL).
Bandwidth ceiling on R9700: 3 / 576 = 5.2 ms. **We're at 7.3× the
ceiling.** Significant headroom.

## Big levers

### A. Q4_K × Q8_1 mmq path  ❌ negative result

**Built and measured. No speedup on Qwen 3.6.** Commit `27f0c76` lands
the `dmmv_q4k_q8_1.comp` shader (mirrors dmmv_q4k.comp's threading,
swaps the inner f32 dot for an i32 dot against pre-quantized Q8_1
activations) and the matching `recordMmqQ4_K_Q8_1` dispatch helper.
Wired into the SSM proj path behind the existing `ZINC_MMQ_SSM=1`
env flag (later commit). With and without the flag on Qwen 3.6:

```
baseline  : SSM phase = 15.94 ms, total decode = 37.73 ms
ZINC_MMQ_SSM=1: SSM phase = 15.94 ms, total decode = 37.74 ms
greedy output: identical token sequence
```

The shader is correct (greedy outputs match bit-for-bit), but the SSM
proj DMMVs are bandwidth-bound on weight reads, not compute-bound on
dequant. Q4_K mmq saves activation bandwidth (4× smaller than f32) and
compute (i32 dot vs f32 unpack-and-multiply), but neither is the
bottleneck on this path on R9700.

**Lesson:** memory-bound matvecs don't benefit from mmq on RDNA4. The
shader is still useful as a foundation for future GEMM-style mmq
(batched / multi-token), where higher arithmetic intensity makes the
integer-dot win actually translate. Keep the pipeline + helper in the
tree.

### B. Batched MoE prefill  (huge, multi-day)

Qwen 3.6 prefill is 27.8 tok/s — essentially identical to decode (25.5
tok/s) because `canUseBatchedPrefillRdna` rejects MoE
(`if (cfg.n_experts > 0) return false`). Every prompt token goes
through the per-token path that re-reads each expert's weights once per
token. For a 100-token prompt × 8 active experts × 30 SSM layers ≈
24 000 expert weight re-reads.

Batched MoE prefill needs to:
1. Run the router for all N prompt tokens up front, collect a
   per-(token, expert-slot) routing table.
2. Group tokens by expert. For each expert, dispatch one big DMMV that
   processes all the tokens routed to it (variable token count per
   expert).
3. Scatter results back into hidden order.

The expert grouping plus variable per-expert batch sizes is the hard
part. Metal already has a similar prefill-batched MoE path; porting
that to Vulkan / RDNA is the work item. Expected gain: 3-5× prefill on
35B-A3B.

### C. SSM proj fusion  (small, low-priority)

SSM proj dispatches 4 separate DMMVs (wqkv, z, alpha, beta) from the
same `norm_buf`. Outputs are different sizes:
- wqkv → conv_channels (~6144)
- z    → d_inner       (~4096)
- alpha → dt_rank      (~32)
- beta  → dt_rank      (~32)

In principle these can run in parallel on the GPU since there are no
dependencies between them and the call site has no inter-DMMV barriers.
On RDNA4 / RADV, separate dispatches in the same command buffer
*should* overlap; profiler `ssm_proj=8.81 ms` is consistent with
either serial dispatch totals OR a parallel run dominated by wqkv.

If they ARE running serially, an alpha+beta 2-way fusion would save
one dispatch per SSM layer × 30 layers ≈ 1.5 ms. If they're parallel,
fusion is a no-op. Worth measuring before investing.

The dense fused gate+up shader landed in commit `339c886` with NUM_ROWS=2
showed +11% regression on Gemma's wide FFN from register pressure — so
any fused SSM proj shader needs to either keep NUM_ROWS=1 (more WGs but
less register pressure) or carefully avoid keeping both nibble tiles
alive at once.

### D. Bigger NUM_ROWS for MoE expert DMMVs  (small)

`dmmv_q4k_wide` (NUM_ROWS=8) helped the LM head when M ≥ 100 000. MoE
expert per-call M is much smaller (per-expert inter_dim ≈ 4096), so
NUM_ROWS=8 won't help there directly. But within an expert's row
range, the kpar shader could amortize the activation reads more
aggressively. Probably fiddly for sub-percent gain.

## Recommended order

1. **A first.** Q4_K × Q8_1 is the biggest single lever. It applies
   to every Q4_K dispatch in the codebase, not just Qwen 3.6 (Gemma
   31B FFN, Qwen 3.5 MoE, etc all benefit).
2. **B next.** Batched MoE prefill is the prefill story — pairs well
   with the existing batched-dense prefill on Gemma / Qwen 3-8B.
3. C and D are not worth pursuing standalone; revisit after A/B.

## Constraints

- Coherence on all 6 RDNA catalog models. Validate via
  `ZINC_BATCHED_PREFILL=validate` for prefill changes and direct
  `--prompt` runs with golden outputs for decode changes.
- Q8_1 quantization noise floor must stay below the `tol=0.001`
  validate threshold on existing models (so the new mmq path defaults
  ON).
- Don't regress Gemma 4 31B prefill / decode (this session's prior
  wins).
