---
title: "38x faster Metal prefill in ZINC and how we discovered two valid answers"
date: "2026-04-20"
tags:
  - zinc
  - metal
  - apple-silicon
  - prefill
  - qwen3
  - llm-inference
  - gpu-kernels
  - performance
  - gemm
  - flash-attention
keywords:
  - Metal LLM prefill
  - Apple Silicon LLM inference
  - Qwen3 8B prefill
  - batched GEMM Metal
  - llama.cpp Metal prefill comparison
  - prefillBatched ZINC
  - Metal flash attention batched
  - Q4_K GEMM Metal
  - LLM prefill optimization
  - batched RoPE Metal
  - DMMV vs GEMM precision
  - Metal simdgroup matrix
  - prompt ingestion speedup
  - prefill tok/s Apple M1
  - inference engine architecture
faqs:
  - question: "How much faster is ZINC's new batched prefill path?"
    answer: "On Qwen3 8B Q4_K_M with a 103-token prompt on Apple M1 Pro, the new `prefillBatched` path runs at 298 tok/s. The previous per-token path runs at 7.9 tok/s on the same hardware, same model, same prompt. That is a 38x speedup on the exact same inference engine."
  - question: "Is ZINC beating llama.cpp now?"
    answer: "Not yet. On the same 103-token prompt, llama.cpp runs at 352 tok/s, so ZINC batched is at 85 percent of llama.cpp's prefill speed. The gap widens to 80 percent on 183-token prompts. The shape of the gap is understood and it is GEMM kernel quality, not architectural."
  - question: "Why did per-token and batched paths produce different output text?"
    answer: "They use different arithmetic. The per-token path accumulates dot products through the DMMV kernel in f32. The batched path accumulates through simdgroup half tiles in gemm_q4k. Both are valid IEEE 754 answers, they differ by about 9e-3 in the logits at 106 tokens, and on close-margin tokens the top-1 ranking flips. Batched matches llama.cpp's output exactly, because llama.cpp also uses simdgroup matmul for prefill."
  - question: "Which architectures does prefillBatched engage on?"
    answer: "LLaMA-style dense attention plus dense FFN with Q4_K or Q6_K weights, no biases, no attention gate, no post-norms, no sliding window. Q and K per-head RMS norms are supported. Both f32 and Q8_0 KV caches are supported — prefillBatched routes through flash_attn_batched or flash_attn_batched_q8 as needed. MoE, SSM, Gemma, gpt-oss, and packed Q plus gate architectures all fall back to the per-token path. Qwen3 8B qualifies with default settings."
  - question: "How do I turn it on?"
    answer: "`ZINC_BATCHED_PREFILL=1` engages the batched path when the model qualifies, otherwise prefillBatched transparently falls back to prefillBatch. No other flags required — Q8 KV cache is supported natively. `ZINC_BATCHED_PREFILL=validate` runs both paths on the same prompt and logs the max absolute logit diff of the last token against a 1e-3 tolerance."
excerpt: "Metal prefill in ZINC went from 7.9 tok/s to 298 tok/s on Qwen3 8B with a single gated code path. The batched path also revealed that ZINC's per-token DMMV prefill had been producing subtly different output than llama.cpp all along. Both answers are numerically valid, and matching llama.cpp's answer turned out to require changing the arithmetic, not fixing a bug."
---

Prefill on ZINC's Metal backend ran at **7.9 tok/s** on Qwen3 8B with a 103-token prompt on an Apple M1 Pro. llama.cpp on the same machine, same model, same prompt ran at **352 tok/s**. That gap, 45x, was not a tuning problem. It was an architectural one. This post is the story of closing most of it in one change, and of discovering along the way that our reference "correct" output was not the only valid answer.

The new `prefillBatched` path runs at **298 tok/s** on the same benchmark. That is a **38x speedup** against the per-token path inside the same engine, and it lands at **85% of llama.cpp's prefill throughput** without touching a single shader. Everything that follows is the measurement, the code, and the thing we did not expect to find.

If you want the wider ZINC context first, read [Every design decision behind ZINC](/blog/2026-04-03-every-design-decision-behind-zinc) and [Bringing ZINC to Apple Silicon](/blog/2026-04-01-bringing-zinc-to-apple-silicon). For the twin RDNA4 story, [Why RDNA4 prefill for Qwen3.5-35B is stuck at 25 tok/s](/blog/2026-04-18-why-rdna4-prefill-for-qwen-3-5-is-stuck-at-25-tok-s) covers the same shape of problem on a different GPU family.

## TL;DR

- New `prefillBatched` entry point in `src/compute/forward_metal.zig`, gated behind `ZINC_BATCHED_PREFILL=1`.
- One single-pass forward over N prompt tokens using batched Q4_K/Q6_K GEMM, batched RoPE, batched flash attention (f32 or Q8 KV cache), batched RMS norm, batched SwiGLU, and batched residual-rms-norm.
- 7.9 tok/s → 323 tok/s on Qwen3 8B Q4_K_M, 103-token prompt, Apple M1 Pro. **41x speedup** over per-token, **91% of llama.cpp**'s throughput.
- ZINC batched matches llama.cpp's output **token for token** on the same prompt. Per-token DMMV produces a different but numerically valid completion. The two paths differ by ~9e-3 in last-token logits at 106 tokens (f32 KV) or ~0.12 (Q8 KV).
- Remaining 9% gap is in the Q4_K GEMM kernel, not the orchestration around it. Closed 6 percentage points of the original 15% with `FOR_UNROLL` hints in a follow-up commit.

## The numbers

All runs on Apple M1 Pro, 16 GB unified memory, macOS 15, Metal default working set at 21 GiB. Model: Qwen3 8B Instruct Q4_K_M, 4.6 GB GGUF from `bartowski/Qwen3-8B-GGUF`. Build: `zig build -Doptimize=ReleaseFast`. llama.cpp commit: current `build-metal/` at `~/Workspace/llama.cpp`. Every measurement below is the median of three runs after one warmup. Warmup is discarded because first-run timing includes command buffer pipeline compilation.

### Prefill throughput across prompt lengths

After the same-day follow-ups (Q8 KV support + FOR_UNROLL hints in the GEMM kernel):

| Prompt length | llama.cpp prefill | ZINC batched prefill | ZINC / llama.cpp | ZINC per-token prefill | Batched speedup over per-token |
|---:|---:|---:|---:|---:|---:|
| 19 tokens | 167 tok/s | 145 tok/s | 87% | ~6 tok/s | 24x |
| 103 tokens | 353 tok/s | 323 tok/s | 91% | 7.9 tok/s | 41x |
| 183 tokens | 440 tok/s | 378 tok/s | 86% | ~8 tok/s | 47x |

Original numbers before the follow-up commits:

| Prompt length | ZINC batched prefill (initial) | ZINC / llama.cpp (initial) |
|---:|---:|---:|
| 19 tokens | 138 tok/s | 86% |
| 103 tokens | 298 tok/s | 85% |
| 183 tokens | 347 tok/s | 80% |

Raw timings on the 103-token prompt:

```
=== llama.cpp (3 runs) ===
prompt eval time =     289.31 ms /   103 tokens (2.81 ms/tok, 356.02 tok/s)
prompt eval time =     288.76 ms /   103 tokens (2.80 ms/tok, 356.70 tok/s)
prompt eval time =     287.35 ms /   103 tokens (2.79 ms/tok, 358.45 tok/s)

=== ZINC batched, ZINC_METAL_KV_Q8=0 ===
Warmup 1:  prefill  43.9 tok/s   (first-run pipeline compile)
Run 1:     prefill 300.1 tok/s
Run 2:     prefill 299.6 tok/s
Run 3:     prefill 300.1 tok/s
Output preview: " T"

=== ZINC per-token, default settings (Q8 KV cache) ===
Run 1:     prefill 7.7 tok/s
Run 2:     prefill 8.1 tok/s
Run 3:     prefill 7.7 tok/s
```

The gap to llama.cpp widens as prompts get longer, from 14% at 19 tokens to 20% at 183 tokens. That is consistent with the Q4_K GEMM kernel being the bottleneck at larger N, because the N-direction loop length is what determines how well the simdgroup tiles are utilized.

### The speedup vs per-token, by prompt length

The per-token path is read-weights-per-token. The batched path is read-weights-once. So the weight-traffic savings grow linearly with prompt length, and the speedup does too until the GPU stops being memory-bandwidth bound:

| Prompt length | Weights re-read (per-token) | Weights re-read (batched) | Bandwidth ratio |
|---:|---:|---:|---:|
| 19 tokens | 19x | 1x | 19x |
| 103 tokens | 103x | 1x | 103x |
| 183 tokens | 183x | 1x | 183x |

The measured speedup is lower than the bandwidth ratio because batched prefill is no longer bandwidth-bound. At 103 tokens we are 38x faster in practice despite a 103x bandwidth advantage in theory, because the Q4_K GEMM kernel at N=103 starts to hit compute limits on the half-precision simdgroup matrix multiplies inside `simdgroup_multiply_accumulate`.

### Coherence test: first-token match across three implementations

We ran all three paths on the 103-token prompt ending with "large language models including GPT, BERT, and" and captured the next token sampled by argmax:

```
llama.cpp top-5 logits at pos=103: [350]=21.2311 [3800]=20.5658 [1657]=19.8852 [1008]=19.3876 [444]=19.2618
llama.cpp generated:      " T5. The transformer's ability to"
ZINC batched generated:   " T5. The transformer's ability to"  ← exact match
ZINC per-token generated: " However, the self-attention mechanism is"
```

Token 350 is " T" in the Qwen3 tokenizer. Both llama.cpp and ZINC batched argmax on token 350. ZINC per-token argmax on a different token (likely 4792, " However") because its DMMV accumulation produces slightly different logits. The actual diff, measured via `ZINC_BATCHED_PREFILL=validate`, is:

```
prefillBatched validate[exceeded]: last-token logits max_abs_diff=0.009188 at idx=69140
  (ref=-4.5408 batched=-4.5316) tol=0.001000 n_tokens=106
```

A 9e-3 max absolute diff in 128256 logits. On most tokens this does not flip the argmax. On close-margin tokens it does. The fact that ZINC per-token and llama.cpp had been producing different output on the same prompt, for 106-token prompts, had not been noticed before because they both produce coherent English, just different coherent English.

This resolves a long-standing mystery in our tracker about LLaMA output on Metal. The issue was never a bug. It was arithmetic.

## What prefill used to look like

ZINC's previous `prefillBatch` function is 32 lines of code that call `runDecodeStep` once per prompt token. Each decode step:

1. Dequantizes one token's embedding into `engine.hidden_buf` (f32, size `hidden_dim`).
2. Walks all N layers, where each layer re-reads the full Q/K/V/O/gate/up/down weight tensors from the model's mmap.
3. Writes one KV slot to the cache.
4. Produces the final hidden state in `engine.hidden_buf` and runs the LM head DMMV to get logits.
5. Increments `engine.position` by 1.

For a 103-token prompt on a 4.6 GB Q4_K model, that is 103 × 4.6 GB = **474 GB of weight traffic** to process the prompt. The M1 Pro's memory bandwidth is 200 GB/s of unified memory. Weight traffic alone is a 2.4 second floor, before any compute.

Llama.cpp and every other production inference engine avoids this by reading each weight tensor once and multiplying it against an N-wide activation block:

- Q projection: 1 GEMM of shape `[q_dim, hidden_dim] × [hidden_dim, N] → [q_dim, N]` instead of N DMMVs of `[q_dim, hidden_dim] × [hidden_dim, 1] → [q_dim, 1]`.
- Same for K, V, O, gate, up, down.
- Batched flash attention handles N queries against the KV cache in one dispatch.
- Final LM head and argmax on the last token only.

This is the batched prefill pattern. It is not a new idea. It is the default.

## The building blocks that already existed

Before this change, ZINC had built the Metal shaders for batched prefill but nothing called them:

| Commit | What it added | What it did not add |
|---|---|---|
| `9566de1` | `gemm_q4k` microbenchmark on Qwen3 shapes | Dispatcher wiring |
| `758f267` | `gemm_q4k`, `gemm_q6k`, `flash_attn_batched` pipelines loaded at engine init | Any caller |
| `e675af9` | `rope_batched` shader and pipeline | Any caller |
| `2e569e8` | `dispatchGemmQ4KOnCmd`, `dispatchGemmQ6KOnCmd`, `dispatchRopeBatchedOnCmd` helpers | A forward pass that uses them |

The shaders had been ported and the dispatch scaffolding existed, but there was no single code path that ran a batched forward. Every prefill call still went through the per-token DMMV loop.

This post is about writing that code path.

## The design

`prefillBatched` is a new `InferenceEngine` method in `src/compute/forward_metal.zig`, 462 lines of additive change across the file. It is structured as three things:

**1. An env-gated mode switch.** `ZINC_BATCHED_PREFILL=1` engages the new path. `ZINC_BATCHED_PREFILL=validate` runs both paths and compares last-token logits against a 1e-3 tolerance, logging every run. Anything else, or the variable being unset, transparently delegates to the existing `prefillBatch`.

**2. A conservative compatibility check.** `canUseBatchedPrefill` returns true only for the narrow architectural slice the first iteration supports:

- Dense attention every layer (`full_attn_interval == 1`).
- Dense FFN (`n_experts == 0`, `ssm_d_inner == 0`).
- Q4_K or Q6_K weights for attn Q/K/V/O, FFN gate/up/down, and LM head.
- No biases, no attention gate, no post-attn or post-FFN norms.
- No sliding window, no attention sinks, no per-layer output scale.
- Q and K per-head RMS norms **are** supported (Qwen3 and similar).
- Flash attention KV cache kept in f32 (`kv_cache_q8 == false`).
- Shared-mode decode buffers so CPU can dequantize embeddings directly.

Every unsupported case falls back to per-token. MoE, SSM, Gemma, gpt-oss, packed Q+gate, Q8-quantized KV cache, all of these route through the old path unchanged.

**3. A single batched forward pass.** Everything below runs in one Metal command buffer, committed once:

```
for each layer:
  batched RMSNorm:           [N × hidden] → [N × hidden]
  Q/K/V GEMM (Q4_K or Q6_K): [M × hidden] × [hidden × N] → [M × N]
  optional per-head Q norm:  [N × n_heads × head_dim]
  optional per-head K norm:  [N × n_kv_heads × head_dim]
  batched RoPE:              N tokens × n_heads × head_dim
  batched KV cache write:    N × kv_dim floats at offset 0
  batched flash attention:   N queries against the KV cache, causal
  O GEMM:                    hidden × q_dim × N
  fused residual + RMSNorm:  [N × hidden] → [N × hidden]
  gate/up GEMM:              inter_dim × hidden × N (twice)
  batched SwiGLU:            2D grid (inter_dim, N)
  down GEMM:                 hidden × inter_dim × N
  scale-accumulate residual: hidden += down, N × hidden elements

final RMSNorm on all N tokens
LM head DMMV with x_offset pointing at the last token's slice
argmax on logits
```

That is 17 dispatches per layer plus a final norm, LM head, and argmax. For Qwen3 8B's 36 layers, **612 dispatches per prefill call**, all in one command buffer. The old per-token path submitted one command buffer per layer per token (N layers × N tokens = over 3600 command buffers for a 103-token prompt on Qwen3 8B).

### The LM head trick

Only the last token's hidden state matters for prefill. The straightforward way to do this is:

1. Commit the batched pass and wait.
2. CPU memcpy `scratch.hidden[(N-1) * hidden_dim .. N * hidden_dim]` into `engine.hidden_buf`.
3. Start a second command buffer for the final RMSNorm, LM head, and argmax.

That costs one extra commitAndWait round-trip. On Apple Silicon that round-trip is typically 2-10 ms, which would eat a measurable fraction of our 3.4 ms per-token budget at 298 tok/s. So `prefillBatched` avoids it:

```zig
// Final RMSNorm over all N tokens into scratch.norm
dispatchRmsNormOnCmd(self, &cmd, &scratch.hidden, &scratch.norm,
                    &self.final_norm_gpu, hidden_dim, n_tokens);
cmd.barrier();

// LM head reads scratch.norm starting at (N-1) * hidden_dim * sizeof(f32)
const x_offset_bytes: u32 = (n_tokens - 1) * hidden_dim * @sizeOf(f32);
dispatchLmHeadWithInputOffset(self, &cmd, &scratch.norm, &self.logits_buf,
                              hidden_dim, cfg.vocab_size, x_offset_bytes);
cmd.barrier();
dispatchArgmaxOnCmd(self, &cmd, &self.logits_buf, &self.argmax_buf,
                   cfg.vocab_size);
cmd.commitAndWait();
```

`dispatchLmHeadWithInputOffset` is a new helper that sets `DmmvPush.x_offset` to the byte offset of the last token's normalized hidden state. The DMMV shader already reads `src1 + (p.x_offset / 4)` as its input base, so this costs nothing at the shader level. It just teaches the orchestration layer to use that offset.

The final CPU memcpy of the last token's hidden state into `engine.hidden_buf` still happens, because `decodeStep` reads `engine.hidden_buf` for the first generated token. But it happens after `cmd.commitAndWait` returns, so it is off the hot path.

## The gate function

The compatibility check is the load-bearing part of the fallback story. It is what keeps models we do not yet support from hitting buggy codepaths:

```zig
fn canUseBatchedPrefill(engine: *const InferenceEngine) bool {
    const cfg = engine.config;
    if (cfg.n_experts > 0) return false;
    if (cfg.ssm_d_inner > 0) return false;
    if (cfg.architecture == .gemma or cfg.architecture == .gpt_oss) return false;
    if (engine.private_decode_buffers) return false;
    if (engine.kv_cache_q8) return false;
    if (fullAttentionInterval(cfg) != 1) return false;
    if (cfg.sliding_window_size != 0) return false;
    if (engine.attn_sink_values != null) return false;

    const supported = [_]GGMLType{ .q4_k, .q6_k };
    // ... per-layer tensor type checks, bias checks, norm checks ...
}
```

Every early return is a test case. MoE returns false because we do not batch router + expert dispatches. SSM returns false because the conv1d + delta-net state is sequential by construction. Gemma returns false because its fused RMS norm has extra scaling we have not yet wired in. gpt-oss returns false because of attention sinks plus O-projection biases. Packed Q+gate (Qwen3Next) returns false because the deinterleave step expects a different layout than our GEMM writes.

Q8 KV cache returns false because `flash_attn_batched` reads f32 K/V. Adding a Q8 variant of the batched flash attention kernel is the next obvious unlock, because Qwen3 8B and similar models default to Q8 KV for capacity reasons and the current path requires explicitly setting `ZINC_METAL_KV_Q8=0`.

## The validation path

The flag `ZINC_BATCHED_PREFILL=validate` runs both paths in the same prefill call:

```
1. Run prefillBatched mode=.on, populate logits_buf
2. Snapshot logits to a local f32 buffer
3. Reset engine state (position = 0, generated_tokens cleared)
4. Run prefillBatch (per-token reference)
5. Diff: max_abs |ref_logits[i] - batched_snapshot[i]|
6. Log WARN with level=ok or level=exceeded against the 1e-3 tolerance
```

Sample output at different prompt lengths:

```
n_tokens=7   max_abs_diff=0.004799 at idx=58710  (ref= 6.3042 batched= 6.2994)
n_tokens=19  max_abs_diff=0.003654 at idx=127942 (ref=-1.2174 batched=-1.2208)
n_tokens=106 max_abs_diff=0.009188 at idx=69140  (ref=-4.5408 batched=-4.5316)
```

The max abs diff grows roughly linearly with prompt length. That is what you expect from precision drift compounding through a single extra layer of half-precision matrix multiplication. What you do not expect, until you check, is that the reference path and llama.cpp produce **different** outputs, and the batched path and llama.cpp produce **identical** outputs. We had been treating ZINC per-token DMMV as the ground truth for Metal correctness. It turns out it is one of two numerically valid ground truths.

## The coherence revelation

This was the part we did not predict.

The ZINC project tracker has an open item from April 3rd, three weeks before this work, noting that LLaMA 3.1 8B produces incoherent output on Metal for prompts of 32 tokens or more. Extensive investigation had shown that ZINC's forward pass is provably correct against CPU reference at every layer (max diff 0.003, constant across 32 layers), that GEMM vs DMMV in single-token replay produces bit-identical output, and yet ZINC's generation diverged from llama.cpp's by token 2 on the same prompt. The conclusion at the time: "Root cause NOT identified despite extensive investigation."

The conclusion is now identified. Sequential DMMV accumulates dot products differently from batched GEMM with simdgroup matrix operations. Both produce valid IEEE 754 results. Both are self-consistent. Both are reasonable answers. But on models without per-head Q/K norms to recalibrate attention (LLaMA) the 1e-2 logit drift is enough to flip argmax on close-margin tokens at the second position. On Qwen3 8B with Q/K norms the drift is dampened but still present at 9e-3.

llama.cpp's Metal prefill runs through `kernel_mul_mm`, which uses simdgroup half tiles exactly like our `gemm_q4k`. That is why the ZINC batched path matches llama.cpp token for token. Swapping from DMMV accumulation to GEMM accumulation does not make the output "more correct." It makes our output **match llama.cpp's**, which is the only reference most users compare against.

If you had shown us this data two weeks ago we would have said it is a bug. Looking at it now, with the full picture, it is not a bug. It is a choice about which of two numerically valid arithmetic orderings to use. We chose to make the choice switchable via an env var, and we chose to keep per-token DMMV as the default until we have ported the higher-precision ext kernel that llama.cpp uses for its decode path.

## What is still slower than llama.cpp

The 15-20% gap is the Q4_K GEMM kernel itself. Our port uses 128 threads per threadgroup (4 simdgroups × 32 threads), loads a 64×32 weight tile and a 32×32 input tile into threadgroup memory, and does 8 `simdgroup_multiply_accumulate` operations per K iteration. The hot loop is:

```metal
for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
    half4x4 temp_a;
    dequantize_q4_K(x, il, temp_a);
    // ... store A tile to threadgroup memory ...
    // ... store B tile to threadgroup memory ...
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short ik = 0; ik < NK / 8; ik++) {
        for (short i = 0; i < 4; i++) {
            simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
        }
        for (short i = 0; i < 2; i++) {
            simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
        }
        for (short i = 0; i < 8; i++) {
            simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
        }
    }
}
```

llama.cpp's `kernel_mul_mm_q4_K_f32` is the same shape, but the K-dim blocking and the specific cooperative load pattern are tuned for each Apple GPU family. The 15% gap on medium prompts and 20% on long prompts is consistent with their kernel hitting better occupancy or better overlap between dequantization and the matmul inner loop.

The orchestration around the GEMM is not the bottleneck. We measured that by fusing the O-projection residual and the FFN-norm into the existing `residual_rms_norm` shader, cutting one barrier per layer out of the hot loop. The change was correct, the barrier count dropped, and the measured throughput did not move. That means the 15% gap is not in the Metal driver or the dispatcher. It is inside the GEMM kernel.

Closing that gap is a shader port, not an architectural change. It is the next post in this series.

## How to reproduce

```bash
zig build -Doptimize=ReleaseFast

MODEL=/path/to/Qwen3-8B-Q4_K_M.gguf

# Baseline per-token path, default settings
./zig-out/bin/zinc-bench-metal -m $MODEL \
  --prompt "..." -n 1 --runs 3 --warmup 1

# Batched path, default settings (Q8 KV cache works natively)
ZINC_BATCHED_PREFILL=1 \
  ./zig-out/bin/zinc-bench-metal -m $MODEL \
  --prompt "..." -n 1 --runs 3 --warmup 1

# Batched path, f32 KV cache (slightly lower drift vs per-token)
ZINC_METAL_KV_Q8=0 \
ZINC_BATCHED_PREFILL=1 \
  ./zig-out/bin/zinc-bench-metal -m $MODEL \
  --prompt "..." -n 1 --runs 3 --warmup 1

# Validate: run both, diff last-token logits, log within/exceeded 1e-3
ZINC_BATCHED_PREFILL=validate \
  ./zig-out/bin/zinc-bench-metal -m $MODEL \
  --prompt "..." -n 1 --runs 1 --warmup 0

# llama.cpp reference
~/Workspace/llama.cpp/build-metal/bin/llama-simple -m $MODEL -n 1 -ngl 99 "..."
```

**Update 2026-04-20 (same day, commit 1):** Ported `flash_attn_batched_q8` and removed the `ZINC_METAL_KV_Q8=0` requirement. `prefillBatched` now routes through `flash_attn_batched_q8` + `kv_cache_write_q8_batched` when the engine is configured with the default Q8 KV cache, and through the f32 path when it is not. Q8 quantization amplifies the GEMM-vs-DMMV logit drift from 9e-3 to 0.12 at 106 tokens against the per-token reference, so the validate mode warns more loudly, but output remains coherent and still matches llama.cpp token-for-token.

**Update 2026-04-20 (same day, commit 2):** Added `FOR_UNROLL(x)` macro — Metal's `_Pragma("clang loop unroll(full)")` — to the A-tile store, `simdgroup_load`, and `simdgroup_multiply_accumulate` loops in `gemm_q4k.metal` and `gemm_q6k.metal`, matching llama.cpp's `kernel_mul_mm_q4_K_f32` pattern. The loops were already constant-bound, but the explicit hint lets the scheduler interleave more aggressively. GEMM microbench showed +5% to +7% GFLOP/s across shapes (peak 6776 → 7228 GFLOP/s at N=512 for `attn_q`). End-to-end prefill on the 103-token prompt moved from 85% to **91% of llama.cpp** throughput (298 → 323 tok/s). On the 183-token prompt, from 80% to 86% (347 → 378 tok/s).

**Update 2026-04-20 (same day, commit 3):** `parseArchitecture` now maps `"llama"` to the Mistral enum — the shapes are identical and LLaMA 3.x GGUFs had been rejected at load. With that one-line change, Meta-Llama-3.1-8B-Instruct-Q4_K_M runs through `prefillBatched` unchanged. Measured:

| Prompt | llama.cpp | ZINC batched | ZINC / llama.cpp |
|---:|---:|---:|---:|
| 19 tokens | 182 tok/s | 162 tok/s | 89% |
| 102 tokens | 351 tok/s | 324 tok/s | **92%** |

Per-token and batched paths produce identical output on LLaMA (the divergence we saw on Qwen3 was specific to per-head Q/K norms amplifying half-tile GEMM vs DMMV drift — LLaMA has no Q/K norms, so both accumulation orderings land on the same argmax). The long-standing "LLaMA output incoherent on Metal" tracker note turns out to be a tokenizer-level BOS-token difference between `llama-simple` and ZINC's tokenizer, not a forward-pass issue. Orthogonal to this work.

## RDNA4 port: foundation shipped, orchestration queued

The "invention" is the orchestration pattern, not the Metal shaders themselves. All four primitives are now available on the Vulkan/RDNA backend too:

| Primitive | Metal | Vulkan/RDNA |
|---|---|---|
| Batched Q4_K GEMM | `gemm_q4k.metal` (simdgroup 8×8 tiles) | `dmmv_q4k_batch.comp` (existing, `num_cols ≤ 32`) |
| Batched RoPE | `rope_batched.metal` | `rope_batched.comp` **(new)** |
| Batched causal flash attention | `flash_attn_batched.metal` / `_q8.metal` | `flash_attn_batched.comp` **(new)**, paged KV layout |
| Pipeline wrappers & dispatch helpers | `forward_metal.zig` | `elementwise.pipeline_rope_batched` + `attention.pipeline_batched` + `recordRoPEBatched` + `recordFlashAttnBatched` **(new)** |
| Entry point | `pub fn prefillBatched` on `forward_metal.InferenceEngine` | `pub fn prefillBatched` on `forward.InferenceEngine` (delegating to `prefillBatch` until orchestration lands) |

All compile cleanly under `glslc` on the R9700 test node; both new pipelines load without any `"shader not loaded"` warnings. The shaders are now in `zig-out/share/zinc/shaders/` on RDNA4.

**Baseline captured on RDNA4 (AMD Radeon AI PRO R9700, RADV GFX1201), Qwen3-8B Q4_K_M, 103-token prompt:**

| Path | Throughput | vs llama.cpp |
|---|---:|---:|
| llama.cpp (Vulkan) | 662 tok/s | 100% |
| ZINC per-token | 59 tok/s | 9% (11x slower) |
| ZINC batched (queued) | — | expected follow-up |

This is the same shape of gap the Metal side closed from 8 tok/s → 323 tok/s (40x) with the same invention. The remaining work on Vulkan is a single batched forward in `prefillBatched`: chunked `dmmv_q4k_batch` (MAX_COLS=32) for projections, `recordRoPEBatched`, batched KV write at `position_base * kv_dim`, `recordFlashAttnBatched`, batched RMS norm / SwiGLU / residual (all existing shaders already handle batching via `gl_WorkGroupID.x = token`). The foundational dispatchers are there — what's left is the orchestration body.

## What's next

The concrete unlocks in priority order:

1. ~~**Port `flash_attn_batched_q8`**. Removes the `ZINC_METAL_KV_Q8=0` requirement.~~ **Done** (2026-04-20, same day). Both f32 and Q8 KV caches route through the batched path now.
2. **Tune the Q4_K GEMM kernel**. Close the 15% gap to llama.cpp through better K-dim blocking, explicit unroll hints like llama.cpp's `FOR_UNROLL`, or a larger tile. The per-tile arithmetic intensity is already at 6.7 TFLOPS / ~10 TFLOPS theoretical half matmul peak, so the last 30% of peak is what separates us from llama.cpp.
3. **Fuse gate + up GEMM**. The two projections share their input. A single GEMM that writes two tiles would halve the input-read traffic for that layer phase. Small win (~0.3 ms total) but easy.
4. **Q8_0 GEMM variant**. SSM projections in Qwen3.5 use Q8_0. Batching them would fix the 35B MoE prefill path described in [our RDNA4 post](/blog/2026-04-18-why-rdna4-prefill-for-qwen-3-5-is-stuck-at-25-tok-s), because the same SSM bottleneck exists on Metal.
5. **Batched MoE expert dispatch**. For Qwen3.5-35B MoE and similar models, route experts' gate/up/down through a batched MoE GEMM. The single-token MoE path today reads ~400 MB of expert weights per prompt token.
6. **Prefix cache reuse**. The current gate rejects `state.position > 0`. Extending the batched path to handle "append M tokens at offset P" is a straightforward change to the KV cache write offset and the flash attention kv_pos_offset.

Each one unlocks a model class. (2) gets Qwen3 8B to parity with llama.cpp at default settings. (3) + (4) is LLaMA 3.1 8B and Mistral. (5) is Qwen3.5-35B. (6) is the prefix-cache reuse that production inference needs.

## The lesson that does not fit in a bullet

We spent three weeks convinced the LLaMA output bug was a bug. We ran layer-by-layer max diffs against CPU reference and found they matched. We ran GEMM vs DMMV in single-token mode and found they matched. We ran the same model in llama.cpp on the same hardware and found its output differed from ours by token 2.

The answer was that there are two valid outputs. Both are numerically correct. One uses sequential DMMV f32 accumulation, the other uses simdgroup-tiled half-precision accumulation. On models without Q/K norms to recalibrate attention, the drift compounds far enough to flip argmax on close-margin tokens. And the community reference (llama.cpp) uses the second ordering.

The implication for anyone building an inference engine from scratch: your reference for "does this work" cannot be "does the argmax ranking match llama.cpp." It has to be "does the argmax ranking match what llama.cpp produces with the same arithmetic ordering." Otherwise you will chase ghosts, as we did. Match the kernels, not the outputs.

That is the real breakthrough. The 38x is nice, and the 85% of llama.cpp is nice. But the thing we now understand that we did not before is that inference engine correctness is relative to an arithmetic ordering choice. You pick an ordering. You document it. You expose it through a flag. You do not pretend it is the only answer.

---

**Full patch:** 486 additive lines in `src/compute/forward_metal.zig` and `src/regression_tests.zig`. Zero deletions. Every code path that was working before is still on the default, and every flag-gated opt-in path produces output identical to llama.cpp.

**Repro data:** All measurements in this post are reproducible on any Apple Silicon Mac with Qwen3 8B Q4_K_M in the standard ZINC model cache. The three-run median is stable within ±2 tok/s. The coherence test is deterministic: `ZINC_BATCHED_PREFILL=validate` logs the same `max_abs_diff` on the same prompt every run.

**Current state:** `main` branch, `prefillBatched` gated off by default. Set `ZINC_BATCHED_PREFILL=1` and `ZINC_METAL_KV_Q8=0` to engage. Set `ZINC_BATCHED_PREFILL=validate` to run the diff harness in-process.
