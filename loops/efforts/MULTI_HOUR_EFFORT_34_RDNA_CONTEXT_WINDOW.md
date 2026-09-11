# Effort 34 — RDNA4 context window: 22K → 262K on one 32 GB card

Qwen 3.8 27B Q4_K_M, AMD R9700 (32,624 MiB, RADV GFX1201), Vulkan backend.
Companion page: https://claude.ai/code/artifact/839cab94-f3eb-4f8d-8fe7-fd063ff7ca96

## Result

| | before | after |
|---|---:|---:|
| default context | 22,016 | 168,960 |
| ceiling | 30,851 | 262,144 (`ZINC_MTP=0`) / 226,934 (MTP on) |
| KV cache per token | 520 KB | 68 KB |

Commits: `28f417f0` (layers + f16 + capture cap), `30c0f9b4` (full window),
`16940d24` (query-tiled attention, opt-in), `d64be7a9` (harness DPM fix).

## The arithmetic

The card holds 32,624 MiB and the weights are ~16 GiB, so context is whatever is
left. The cache was costing 4x what the model needs and storing it at double
width:

- **Only 17 of 65 layers attend.** Full attention runs every 4th layer (16 of
  64) plus the appended NextN block. The Vulkan engine allocated a cache for all
  65; the 48 DeltaNet layers keep recurrent state and never read it. The Metal
  backend already allocated correctly (`isFullAttentionLayer`) but the *shared*
  planner counted every layer, so Metal was under-sizing its context too.
  `memory_plan.kvLayerCount()` now teaches both. → 3.82x
- **f32 → f16.** `compute/kv_dtype.zig` decides once at init; `shaderName()`
  swaps in `_f16kv` shader variants so no dispatch site changed. → 2x
- **The NextN block's 276 MiB of weights** are not uploaded when MTP is off
  (`loader.isAppendedNextnTensor`). That is the last 266 MiB between 257,884 and
  the full 262,144. → +4,260 tokens

The f16 conversion is mechanical because element indices are in units of 4
elements, never bytes: an `f16vec4` holds the same four elements a `vec4` did, so
every base/offset computation carries over verbatim. Readers wrap loads in
`vec4(...)`, writers wrap stores in `float16_t(...)`.

## Two bugs that cost the most time

**Every cache writer must be converted or it corrupts silently.** There are five
(`kv_cache_write`, `kv_cache_write_batched`, `qk_norm_rope_kv_write{,_batched}`,
`k_norm_rope_kv_write_batched`) and two readers (`flash_attn{,_batched}`). The
single-token `kv_cache_write` was missed. It serves the MoE decode path *and
every NextN draft step*, so Qwen3.6 35B A3B produced wrong tokens outright while
the 27B only lost 9 points of draft acceptance (54% → 45%) — which reads exactly
like "f16 is imprecise" and is not. With all five converted, output is
bit-identical to f32 on both models.

**Both VRAM accountants walked the GGUF tensor list, not the loaded tensors.**
`forward.tensorBytes()` and `model_manager.tensorBytes()` summed
`model.gguf_file.tensors.items`, so the 276 MiB the loader had just skipped
stayed subtracted from the KV budget and the context did not move at all. They
walk `model.tensors` now.

## Speculative decoding at depth

The prime capture buffer was capped at 2,048 rows, so any longer prompt silently
fell back to ordinary decode. The cliff is sharp: **78.2 tok/s at a 1,568-token
prompt, 34.3 at 2,467.** Cap is 32,768 now (`ZINC_MTP_CAPTURE_ROWS`), which costs
671 MB of reserve — about 5K tokens of context — and is what makes speculation
reach the depths the bigger window opens up.

At 262,144 speculation cannot fit: the NextN block is a 17th attending layer and
its cache is a full GiB at that length. That is the whole reason `ZINC_MTP=0` is
required for the full window.

## Against llama.cpp (same card, same GGUF, same driver)

| | ZINC | llama.cpp |
|---|---:|---:|
| max context | 262,144 | 262,144 |
| cache per token | 68 KB | ~64 KB |
| decode, short prompt | **55.0** | 31.0 |
| decode, 20K context | **44.2** | 28.9 |
| decode, 262K context | **33.9** | 30.5 |
| prefill, 20K context | 362.7 | 459.3 |

Both prefill numbers are clean (idle card, single request, 2026-09-10). The
sections at the end of this log carry the decomposition.

## Open: prefill

A 20K prefill costs 59.6 s — dense FFN 23.2 s, DeltaNet 19.0 s, attention 16.9 s.
Converted to arithmetic throughput:

- dense FFN gate/up (Q4_K): **32.5 TFLOP/s** — healthy
- dense FFN down (Q6_K): 25.1 TFLOP/s
- DeltaNet QKV (Q6_K): **12.2 TFLOP/s** — the outlier, and note the dense down is
  also Q6_K, so this is that specific path, not the quant format
- DeltaNet out-proj: 13.2 TFLOP/s
- attention: **4.7 TFLOP/s**

Attention is structurally wrong for prefill: `flash_attn_batched` dispatches one
workgroup per (head, query) and each walks its single query's keys alone, so a
loaded K vector produces exactly one dot product and the 6 query heads sharing a
KV head each re-read the same keys. `flash_attn_batched_qt` (committed, opt-in
via `ZINC_FA_QUERY_TILE=1`) gives one workgroup 4 consecutive queries so each
loaded K/V row feeds 4 dot products.

**It has no clean A/B yet** — the node was serving a model during every attempt,
so a 48-token prompt read 158 vs 161 tok/s, which measures the contention. Also
note its output is *not* bit-identical to the untiled kernel and is not supposed
to be: per-query arithmetic is the same but summed serially rather than through
the untiled kernel's 16-way tree. Judge it on needle retrieval and speed.

Rough budget to overtake 466 tok/s: attention at 15 TFLOP/s saves ~11 s and the
Q6_K projections at Q4_K efficiency save ~7 s, which is 59.6 → ~42 s = 480 tok/s.

## Measurement gotchas (both cost a wrong conclusion this session)

- **The perf suite forced `power_dpm_force_performance_level=high`,** which locks
  a nominal DPM state and gives up boost: 4.1–4.4% slower for ZINC and 3.3–3.5%
  for llama.cpp, on every historical RDNA number. Fixed in `d64be7a9`; the pin
  came from the 9070 XT, a different card whose SCLK idled at 0 MHz.
- **The RDNA target benchmarks the server over the OpenAI API, not the CLI,** so
  its prompts carry ~19 chat-template tokens, which lowers draft acceptance. A
  published row reads ~1 tok/s under the same text through `--chat --prompt`.

## Running it

| launch | context | decode | speculation |
|---|---:|---:|---|
| no `-c` | 168,960 | ~55 | on |
| `-c 226934` | 226,934 | ~55 | on |
| `ZINC_MTP=0 -c 262144` | 262,144 | ~34 | off |

The long-lived 9090 server on the node is pinned at 32k by a launcher invoked
over SSH from outside the node — nothing on the node carries that flag and the
process tree ends at sshd, so every respawn returns at 32k until that command
changes.

Switches: `ZINC_KV_F16=0` (f32 cache), `ZINC_MTP=0`, `ZINC_MTP_CAPTURE_ROWS`,
`ZINC_FA_QUERY_TILE=1`.

Tests at the end of this effort: Zig 617/618 (1 skip), bun 639 pass / 3 skip.

## Prefill, measured 2026-09-10 (idle card, 20,043-token prompt)

The card was finally free, so the prefill work has real numbers. Both of my
predictions were wrong in instructive ways.

| | 20K prefill |
|---|---:|
| unchunked (previous behaviour) | 336.7 tok/s |
| chunked, 1 GB scratch budget | 352.6 |
| chunked, 2 GB | 356.1 |
| chunked + query-tiled attention | 255.6 |
| **llama.cpp, clean single request** | **459.3** |

**llama.cpp's prefill is 459 tok/s** — 19,934 tokens in 43.4 s on an idle card.
The earlier 165 tok/s reading was contention from my own profiling job; the 466
reading was genuine. Quote 459.

**Chunking is worth ~5%, not the fix I claimed.** The scratch spill was real
(6.1 GiB wanted, 5.1 GiB free) but it was never the dominant cost. Re-profiling
with chunking on shows why: attention is now ~43% of prefill (was 28%), dense FFN
~35%, DeltaNet ~21%. Chunking cut the linear work; attention is quadratic and
untouched by it, so its share grew.

**The query-tiled attention kernel is a 28% regression** (356 -> 256) and stays
off. The reuse argument was right and the implementation was wrong: it spends
~21 KB of LDS per workgroup against the original's ~4 KB, which collapses
occupancy, and it has 4 accumulator chains where the original unrolls 16. Memory
reuse bought at the price of latency hiding is a losing trade on this part.

**What a correct attention kernel needs:** keep the 16-way ILP and small LDS
while getting reuse. The promising shape is grouping the `q_per_kv` = 6 query
heads that share a KV head (one workgroup = 1 query position x 6 heads): the same
K load feeds 6 dot products, Q costs 6 KB of LDS instead of 16, and the output
accumulators stay in registers (6 vec4 = 24 VGPRs per lane) rather than LDS.
Budget: attention is ~24 s of the 57 s; llama.cpp is 13.6 s ahead, so attention
needs roughly a 2.2x speedup to close the gap on its own.

**Test hygiene note:** the needle assertion in the first A/B reported MISS on
every run because it only generated 8 tokens — not enough for the model to
answer. It proved nothing. Generate >= 40 before trusting it.

## Where the prefill gap actually is (2026-09-10, measured)

Three attention rewrites, all measured on an idle card against a 20,043-token
prompt:

| attempt | hypothesis | result |
|---|---|---:|
| query tiling (4 queries/workgroup) | bandwidth, via query reuse | −28% |
| head grouping, 64 threads | bandwidth, via 6:1 GQA reuse | **+2.8%** |
| head grouping, 384 threads | wave occupancy | −7% |

Cutting K/V traffic sixfold moved the total 2.8%; raising waves-in-flight
sixfold made it worse. Only the middle one is committed (`1d3975e9`).

**Do not trust the prefill phase profiler for attribution.** With grouping on it
reports attention *slower* (1.87 ms/token vs 1.66) while the pass is *faster*
(55.4 s vs 56.4), and its per-phase figures sum to 3.83 ms/token against an
actual 2.81. The "attention is 43% of prefill" figure it produced is what sent me
into three kernels; the real share is 22%.

**The honest decomposition** comes from the length curve, since attention is the
only quadratic term. Chunking and grouping on:

| prompt | ms/token |
|---:|---:|
| 4,099 | 2.282 |
| 8,086 | 2.377 |
| 16,040 | 2.613 |
| 20,043 | 2.777 |

Least squares on T/n = a + b·n gives a = 2.141 ms/token and b = 3.08e-5
ms/token², predicting 55.3 s against 55.7 measured. At 20K that is:

- **linear (projections, FFN, DeltaNet): 42.9 s, 78%**
- **quadratic (attention): 12.4 s, 22%**

llama.cpp does the whole 20K pass in 43.4 s = 2.177 ms/token. **Our linear term
alone is 2.141 ms/token** — essentially their entire pass. So the per-token work
is at parity and the whole 12.3 s gap is attention: 79 TFLOP in 12.4 s = 6.4
TFLOP/s, against the ~32 TFLOP/s the dense FFN reaches on the same silicon.

**What that means for the next attempt.** Attention needs ~5x, not 20%, so
variants of the current shape cannot get there — the three above moved it by at
most 13% of the phase. The shape itself is the problem: one workgroup per query
position walking its whole causal range. A FlashAttention-2 style kernel tiles a
block of queries against a block of keys, keeps the accumulators in registers
across the key loop, and stages K/V tiles in LDS once per query block. That is a
focused piece of work with a clear target (6.4 -> ~30 TFLOP/s), and it is the
only remaining item between us and llama.cpp on prefill.

## Both sides decomposed (2026-09-11)

The same length-curve fit on llama.cpp, same prompts, idle card:

| at 20K | ZINC | llama.cpp |
|---|---:|---:|
| linear (per-token) | 2.141 ms/tok → 42.8 s | 1.956 ms/tok → 39.1 s |
| quadratic (attention) | 3.08e-5 → 12.3 s | 0.99e-5 → 4.0 s |

So their attention is 3.1x faster, not free, and the 12 s gap is **8.3 s of
attention plus 3.7 s of linear work**. Matching their attention alone lands at
46.8 s — still 3.7 s behind. Beating them needs attention *and* ~9% off the
linear term.

The linear side: the DeltaNet QKV projection already runs the DP4a Q8_1 path in
prefill (`qwenDenseSsmProjDp4aEnabled`, n_tokens >= 32), so its 12.2 TFLOP/s
against the dense down's 25.1 on the same Q6_K kernel is efficiency at K=5120
versus K=17408 — a per-tile overhead that a 3.4x shorter K loop cannot amortize —
not a missing switch.

**The reference scalar attention kernel's shape for head size 256 on AMD**
(derived from its tuning selection): workgroup 256 = 4 independent wave64s,
each wave owning Br/row_split = 4 query rows against Bc = 32 key columns per
step, head dim split 8 ways across lanes (each lane holds a 32-dim slice), no
LDS staging of K/V, the 4x4 score tile and the 4x8-vec4 output tile in
registers, and — the part that matters — **each 8-lane column group keeps its
own online softmax** so the inner loop has no cross-lane reduction; the 8
partials merge once at the end through shuffles. Per K vec4 loaded: 4 FMAs
(rows); per Q vec4 read: 4 FMAs (columns). `flash_attn_batched_tile.comp` is
that shape on our paged f16 cache with causal masking done arithmetically and
GQA heads packed as rows (r = q·gqa + h) so heads sharing a KV head share a wave.

**Result of that shape** (`flash_attn_batched_tile`, 20,043-token prompt, idle
card, both runs retrieve the needle):

| | 20K prefill |
|---|---:|
| head-grouped kernel (previous best) | 360.2 tok/s |
| **register-tiled kernel** | **397.5** |
| llama.cpp | 459.3 |

55.6 s → 50.4 s: attention from ~12.3 s to ~7.1 s (1.7x). Gap to llama.cpp is
1.16x, from 1.29x. First attention change of four that moved the needle, and it
is the one that changed the algorithm's shape rather than its memory traffic or
occupancy.

